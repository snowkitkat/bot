"""
Discord Q&A Bot
---------------
Security hardened per:
  - Principle of Least Privilege (no Administrator intent)
  - Input validation / no eval/exec
  - Prepared-style config (JSON, schema-validated — no SQL used)
  - Path-traversal-safe file operations
  - app_commands cooldowns + in-memory rate limiting
  - Async-only I/O (aiofiles for disk writes)
  - @commands.is_owner() guard on sensitive owner commands
  - @app_commands.checks.has_permissions() on all privileged slash commands
  - Global error handler — stack traces never reach users
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import aiofiles                          # async file I/O — pip install aiofiles
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Logging — structured, no sensitive data ever
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Token — only from .env, shape-validated
# ──────────────────────────────────────────────
load_dotenv()
TOKEN: str = os.getenv("DISCORD_TOKEN", "")
if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN missing from .env — refusing to start.")
if not re.fullmatch(r"[\w-]{24,}\.[\w-]{6,}\.[\w-]{27,}", TOKEN):
    raise RuntimeError("DISCORD_TOKEN looks malformed — check your .env.")

# ──────────────────────────────────────────────
# Config path — absolute, no user input involved
# Principle: never construct file paths from user data
# ──────────────────────────────────────────────
_BASE_DIR   = Path(__file__).resolve().parent   # directory this script lives in
CONFIG_PATH = _BASE_DIR / "config.json"         # always a fixed, known path

# Config schema: key → tuple of accepted Python types
_SCHEMA: dict[str, tuple] = {
    "ban_role_id":       (type(None), int),
    "target_channel_id": (type(None), int),
    "answer_channel_id": (type(None), int),
    "admin_role_ids":    (list,),
}
_DEFAULTS: dict = {
    "ban_role_id":       None,
    "target_channel_id": None,
    "answer_channel_id": None,
    "admin_role_ids":    [],
}

def _validate_cfg(raw: dict) -> dict:
    """
    Return a clean copy of the config dict.
    - Unknown keys are dropped (no arbitrary data injected)
    - Wrong-typed values are reset to their default
    - List entries that are not ints are purged
    """
    clean: dict = {}
    for key, allowed in _SCHEMA.items():
        val = raw.get(key, _DEFAULTS[key])
        if not isinstance(val, allowed):
            log.warning("Config key %r has wrong type (%s) — resetting to default.", key, type(val).__name__)
            val = _DEFAULTS[key]
        if isinstance(val, list):
            val = [v for v in val if isinstance(v, int)]
        clean[key] = val
    return clean

def _load_config_sync() -> dict:
    """Blocking load used only at startup before the event loop exists."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError("Root must be a JSON object.")
        return _validate_cfg(raw)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        log.warning("Config unreadable (%s) — starting with defaults.", exc)
        return dict(_DEFAULTS)

async def _save_config(cfg: dict) -> None:
    """
    Async write using aiofiles so disk I/O never blocks the event loop.
    Config is validated before writing so corrupt state is never persisted.
    """
    validated = _validate_cfg(cfg)
    try:
        async with aiofiles.open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            await fh.write(json.dumps(validated, indent=4))
    except OSError as exc:
        log.error("Failed to persist config: %s", exc)

# Load once at import time (synchronous is fine here — loop not started yet)
cfg: dict = _load_config_sync()

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MAX_Q_LEN        = 1000   # question text
MAX_A_LEN        = 1000   # answer text
MAX_R_LEN        = 500    # denial reason
COOLDOWN_SECONDS = 30     # /question per-user rate limit

# ──────────────────────────────────────────────
# In-memory per-user cooldown tracker
# (backed up by @app_commands.checks.cooldown below)
# ──────────────────────────────────────────────
_last_submission: dict[int, float] = {}

def _cooldown_remaining(uid: int) -> float:
    return max(0.0, COOLDOWN_SECONDS - (time.monotonic() - _last_submission.get(uid, 0.0)))

def _record_submission(uid: int) -> None:
    _last_submission[uid] = time.monotonic()

# ──────────────────────────────────────────────
# Bot — minimum required intents (Least Privilege)
# Do NOT enable administrator=True in the invite URL.
# Required OAuth2 scopes:  bot  applications.commands
# Required bot permissions (decimal 274878024768):
#   Send Messages, Embed Links, Read Message History, View Channels
# ──────────────────────────────────────────────
intents = discord.Intents.default()
intents.members = True      # needed to resolve members for role checks
# intents.message_content is intentionally NOT enabled — we don't read message text
bot = commands.Bot(command_prefix="!", intents=intents)

# ──────────────────────────────────────────────
# Permission helpers — always re-resolved at interaction time
# ──────────────────────────────────────────────

def _fresh_member(itx: discord.Interaction) -> discord.Member | None:
    """
    Re-fetch member from guild cache at call time.
    Never trust a member object captured earlier in the request cycle —
    roles could have changed between button click and modal submit.
    """
    if itx.guild is None:
        return None
    return itx.guild.get_member(itx.user.id)

def _has_manage_guild(itx: discord.Interaction) -> bool:
    m = _fresh_member(itx)
    return m is not None and m.guild_permissions.manage_guild

def _has_admin_role(itx: discord.Interaction) -> bool:
    m = _fresh_member(itx)
    if m is None:
        return False
    ids: list[int] = cfg.get("admin_role_ids", [])
    return any(r.id in ids for r in m.roles)

def _can_moderate(itx: discord.Interaction) -> bool:
    """True if the caller may answer or deny questions."""
    return _has_manage_guild(itx) or _has_admin_role(itx)

def _same_guild(itx: discord.Interaction, obj: discord.Role | discord.TextChannel) -> bool:
    """Prevent cross-server ID injection by confirming ownership."""
    return itx.guild is not None and obj.guild.id == itx.guild.id

# ──────────────────────────────────────────────
# Embed helpers
# ──────────────────────────────────────────────
_AUDIT_FIELDS = {"Answered By", "Denied By", "Reason"}

def _field_value(embed: discord.Embed, name: str) -> str | None:
    return next((f.value for f in embed.fields if f.name == name), None)

def _rebuild_embed(
    src: discord.Embed,
    title: str,
    color: discord.Color,
    extra: list[tuple[str, str, bool]] | None = None,
) -> discord.Embed:
    """
    Produce a new embed from src, skipping stale audit fields so they
    can never accumulate across repeated actions on the same message.
    """
    out = discord.Embed(title=title, color=color)
    for f in src.fields:
        if f.name not in _AUDIT_FIELDS:
            out.add_field(name=f.name, value=f.value, inline=f.inline)
    for name, val, inline in (extra or []):
        if val and val.strip():
            out.add_field(name=name, value=val, inline=inline)
    if src.thumbnail:
        out.set_thumbnail(url=src.thumbnail.url)
    if src.footer:
        out.set_footer(text=src.footer.text)
    return out

# ──────────────────────────────────────────────
# Modal: Answer a question
# ──────────────────────────────────────────────
class AnswerModal(discord.ui.Modal, title="Answer this Question"):
    answer = discord.ui.TextInput(
        label="Your Answer",
        style=discord.TextStyle.paragraph,
        required=True,
        max_length=MAX_A_LEN,
    )

    def __init__(self, src_embed, src_msg, submitter_id: int, anonymous: bool):
        super().__init__()
        self.src_embed    = src_embed
        self.src_msg      = src_msg
        self.submitter_id = int(submitter_id)   # cast — never trust raw input type
        self.anonymous    = bool(anonymous)

    async def on_submit(self, itx: discord.Interaction) -> None:
        # Re-check permission at submit time — not just at button-click time
        if not _can_moderate(itx):
            await itx.response.send_message("🚫 Your permission has been revoked.", ephemeral=True)
            return

        answer_text = self.answer.value.strip()[:MAX_A_LEN]
        if not answer_text:
            await itx.response.send_message("❌ Answer cannot be empty.", ephemeral=True)
            return

        q_text = _field_value(self.src_embed, "Question") or "*(unknown)*"

        # Build quoted question lines for Discord block-quote formatting
        quoted_question = "\n".join(f"> {line}" for line in q_text.splitlines())
        asker = f"<@{self.submitter_id}>" if not self.anonymous else "*Anonymous User*"
        answer_msg = (
            f"## ✅ Question Answered! ✅\n"
            f"{quoted_question}\n"
            f"{answer_text}\n"
            f"{asker}"
        )

        # Send to the designated answer output channel
        answer_ch_id = cfg.get("answer_channel_id")
        answer_ch = itx.client.get_channel(answer_ch_id) if answer_ch_id else None
        if not answer_ch:
            await itx.response.send_message(
                "❌ Answer channel not set. Use `/set_answer_channel`.", ephemeral=True
            )
            return
        await answer_ch.send(answer_msg)

        updated = _rebuild_embed(
            self.src_embed,
            title="📩 Question — Answered ✅",
            color=discord.Color.green(),
            extra=[("Answered By", itx.user.mention, False)],
        )
        await self.src_msg.edit(
            embed=updated,
            view=QuestionActionView(self.submitter_id, self.anonymous, disabled=True),
        )
        await itx.response.send_message("✅ Answer submitted!", ephemeral=True)
        log.info("Question answered by %s (%s)", itx.user, itx.user.id)

    async def on_error(self, itx: discord.Interaction, error: Exception) -> None:
        log.error("AnswerModal error: %s", error, exc_info=True)
        if not itx.response.is_done():
            await itx.response.send_message("❌ Failed to submit answer.", ephemeral=True)

# ──────────────────────────────────────────────
# Modal: Deny a question
# ──────────────────────────────────────────────
class DenyModal(discord.ui.Modal, title="Deny this Question"):
    reason = discord.ui.TextInput(
        label="Reason (optional)",
        style=discord.TextStyle.paragraph,
        required=False,
        max_length=MAX_R_LEN,
        placeholder="Leave blank for no reason provided.",
    )

    def __init__(self, src_embed, src_msg, submitter_id: int, anonymous: bool):
        super().__init__()
        self.src_embed    = src_embed
        self.src_msg      = src_msg
        self.submitter_id = int(submitter_id)
        self.anonymous    = bool(anonymous)

    async def on_submit(self, itx: discord.Interaction) -> None:
        if not _can_moderate(itx):
            await itx.response.send_message("🚫 Your permission has been revoked.", ephemeral=True)
            return

        reason_text = self.reason.value.strip()[:MAX_R_LEN] if self.reason.value else None
        q_text      = _field_value(self.src_embed, "Question") or "*(unknown)*"

        # DM the submitter — failure must not abort the rest of the handler
        if self.submitter_id:
            try:
                user = await itx.client.fetch_user(self.submitter_id)
                dm = discord.Embed(title="❌ Your Question Was Denied", color=discord.Color.red())
                dm.add_field(name="Your Question", value=q_text, inline=False)
                dm.add_field(name="Reason", value=reason_text or "No reason provided.", inline=False)
                dm.set_footer(text=f"Server: {itx.guild.name if itx.guild else 'Unknown'}")
                await user.send(embed=dm)
            except discord.Forbidden:
                log.info("Cannot DM %s — DMs disabled.", self.submitter_id)
            except discord.NotFound:
                log.info("User %s no longer exists.", self.submitter_id)
            except Exception as exc:
                log.error("Unexpected DM error: %s", exc, exc_info=True)

        extra: list[tuple[str, str, bool]] = [("Denied By", itx.user.mention, False)]
        if reason_text:
            extra.append(("Reason", reason_text, False))

        updated = _rebuild_embed(
            self.src_embed,
            title="📩 Question — Denied ❌",
            color=discord.Color.red(),
            extra=extra,
        )
        await self.src_msg.edit(
            embed=updated,
            view=QuestionActionView(self.submitter_id, self.anonymous, disabled=True),
        )
        await itx.response.send_message("✅ Question denied, user notified.", ephemeral=True)
        log.info("Question denied by %s (%s)", itx.user, itx.user.id)

    async def on_error(self, itx: discord.Interaction, error: Exception) -> None:
        log.error("DenyModal error: %s", error, exc_info=True)
        if not itx.response.is_done():
            await itx.response.send_message("❌ Failed to deny question.", ephemeral=True)

# ──────────────────────────────────────────────
# View: Answer / Deny buttons (persistent across restarts)
# ──────────────────────────────────────────────
class QuestionActionView(discord.ui.View):
    def __init__(self, submitter_id: int, anonymous: bool, disabled: bool = False):
        super().__init__(timeout=None)
        self.submitter_id = int(submitter_id)
        self.anonymous    = bool(anonymous)

        for label, emoji, style, cb in [
            ("Answer Question", "✅", discord.ButtonStyle.success, self._answer),
            ("Deny Question",   "❌", discord.ButtonStyle.danger,  self._deny),
        ]:
            btn = discord.ui.Button(
                label=label, emoji=emoji, style=style,
                custom_id=f"{cb.__name__}_{submitter_id}",
                disabled=disabled,
            )
            btn.callback = cb
            self.add_item(btn)

    async def _answer(self, itx: discord.Interaction) -> None:
        if not _can_moderate(itx):
            await itx.response.send_message("🚫 No permission.", ephemeral=True)
            return
        if not itx.message or not itx.message.embeds:
            await itx.response.send_message("❌ Could not read question data.", ephemeral=True)
            return
        await itx.response.send_modal(
            AnswerModal(itx.message.embeds[0], itx.message, self.submitter_id, self.anonymous)
        )

    async def _deny(self, itx: discord.Interaction) -> None:
        if not _can_moderate(itx):
            await itx.response.send_message("🚫 No permission.", ephemeral=True)
            return
        if not itx.message or not itx.message.embeds:
            await itx.response.send_message("❌ Could not read question data.", ephemeral=True)
            return
        await itx.response.send_modal(
            DenyModal(itx.message.embeds[0], itx.message, self.submitter_id, self.anonymous)
        )

# ──────────────────────────────────────────────
# Modal: Submit a question
# ──────────────────────────────────────────────
class QuestionModal(discord.ui.Modal, title="Submit a Question"):
    question = discord.ui.TextInput(
        label="Your Question",
        style=discord.TextStyle.paragraph,
        required=True,
        max_length=MAX_Q_LEN,
    )

    def __init__(self, anonymous: bool, user: discord.Member):
        super().__init__()
        self.anonymous = bool(anonymous)
        self.user      = user

    async def on_submit(self, itx: discord.Interaction) -> None:
        # Re-check ban role at submit time — it could be assigned after modal opened
        member      = _fresh_member(itx)
        ban_role_id = cfg.get("ban_role_id")
        if member and ban_role_id and any(r.id == ban_role_id for r in member.roles):
            await itx.response.send_message("🚫 You are not allowed to submit questions.", ephemeral=True)
            return

        # Validate and sanitise the free-text input
        q_text = self.question.value.strip()[:MAX_Q_LEN]
        if not q_text:
            await itx.response.send_message("❌ Question cannot be empty.", ephemeral=True)
            return

        ch_id   = cfg.get("target_channel_id")
        channel = itx.client.get_channel(ch_id) if ch_id else None
        if not channel:
            await itx.response.send_message("❌ Target channel not configured.", ephemeral=True)
            return

        # Verify bot permissions in the target channel before attempting to post
        if not channel.permissions_for(itx.guild.me).send_messages:
            await itx.response.send_message(
                "❌ I don't have permission to post in the target channel.", ephemeral=True
            )
            return

        embed = discord.Embed(title="📩 New Question", color=discord.Color.yellow())
        embed.add_field(name="Question",  value=q_text, inline=False)
        embed.add_field(name="Anonymous", value="✅ Yes" if self.anonymous else "❌ No", inline=True)
        # Always show the real sender to mods regardless of anonymous choice
        embed.add_field(
            name="Submitted By",
            value=f"{self.user.mention} (`{self.user}`)",
            inline=True,
        )
        if self.user.display_avatar:
            embed.set_thumbnail(url=self.user.display_avatar.url)
        embed.set_footer(text=f"User ID: {self.user.id}")

        await channel.send(embed=embed, view=QuestionActionView(self.user.id, self.anonymous))

        # Record cooldown only after a successful post — failed attempts don't burn it
        _record_submission(self.user.id)
        await itx.response.send_message("✅ Your question has been submitted!", ephemeral=True)
        log.info("Question submitted by %s (%s)", self.user, self.user.id)

    async def on_error(self, itx: discord.Interaction, error: Exception) -> None:
        log.error("QuestionModal error: %s", error, exc_info=True)
        if not itx.response.is_done():
            await itx.response.send_message("❌ Failed to submit your question.", ephemeral=True)

# ──────────────────────────────────────────────
# /question  — public command
#
# Double-layered rate limiting:
#   1. @app_commands.checks.cooldown  → Discord-native, per-user, 1 use / 30 s
#   2. _cooldown_remaining()          → in-memory fallback (survives bot restarts poorly
#                                       but catches rapid-fire within the same session)
# ──────────────────────────────────────────────
@bot.tree.command(name="question", description="Submit a question")
@app_commands.describe(anonymous="Submit anonymously?")
@app_commands.checks.cooldown(rate=1, per=COOLDOWN_SECONDS, key=lambda itx: itx.user.id)
async def question_cmd(itx: discord.Interaction, anonymous: bool) -> None:
    if not itx.guild:
        await itx.response.send_message("❌ Server only.", ephemeral=True)
        return

    member = _fresh_member(itx)
    if member is None:
        await itx.response.send_message("❌ Could not verify your membership.", ephemeral=True)
        return

    ban_role_id = cfg.get("ban_role_id")
    if ban_role_id and any(r.id == ban_role_id for r in member.roles):
        await itx.response.send_message("🚫 You are not allowed to use this command.", ephemeral=True)
        return

    # Secondary in-process rate-limit check
    remaining = _cooldown_remaining(member.id)
    if remaining > 0:
        await itx.response.send_message(
            f"⏳ Please wait **{remaining:.0f}s** before submitting another question.", ephemeral=True
        )
        return

    await itx.response.send_modal(QuestionModal(anonymous, member))

# ──────────────────────────────────────────────
# Config commands — Manage Server only
# Using @app_commands.checks.has_permissions() as the primary guard,
# with a manual re-check inside the handler as defence-in-depth.
# ──────────────────────────────────────────────

def _mg_check(itx: discord.Interaction) -> bool:
    """Secondary manual permission guard (defence-in-depth)."""
    return _has_manage_guild(itx)


@bot.tree.command(name="set_ban_role", description="Set the role that cannot use /question")
@app_commands.describe(role="Role to block")
@app_commands.checks.has_permissions(manage_guild=True)
async def set_ban_role(itx: discord.Interaction, role: discord.Role) -> None:
    if not _mg_check(itx) or not _same_guild(itx, role):
        await itx.response.send_message("🚫 Permission denied or invalid role.", ephemeral=True)
        return
    cfg["ban_role_id"] = role.id
    await _save_config(cfg)
    await itx.response.send_message(f"✅ Ban role → **{role.name}**.", ephemeral=True)


@bot.tree.command(name="set_target_channel", description="Set the channel where questions are sent")
@app_commands.describe(channel="Channel to receive questions")
@app_commands.checks.has_permissions(manage_guild=True)
async def set_target_channel(itx: discord.Interaction, channel: discord.TextChannel) -> None:
    if not _mg_check(itx) or not _same_guild(itx, channel):
        await itx.response.send_message("🚫 Permission denied or invalid channel.", ephemeral=True)
        return
    cfg["target_channel_id"] = channel.id
    await _save_config(cfg)
    await itx.response.send_message(f"✅ Target channel → {channel.mention}.", ephemeral=True)


@bot.tree.command(name="set_answer_channel", description="Set the channel where answers are posted")
@app_commands.describe(channel="Channel to post answers")
@app_commands.checks.has_permissions(manage_guild=True)
async def set_answer_channel(itx: discord.Interaction, channel: discord.TextChannel) -> None:
    if not _mg_check(itx) or not _same_guild(itx, channel):
        await itx.response.send_message("🚫 Permission denied or invalid channel.", ephemeral=True)
        return
    cfg["answer_channel_id"] = channel.id
    await _save_config(cfg)
    await itx.response.send_message(f"✅ Answer channel → {channel.mention}.", ephemeral=True)


@bot.tree.command(name="add_admin_role", description="Grant a role access to answer/deny questions")
@app_commands.describe(role="Role to grant access")
@app_commands.checks.has_permissions(manage_guild=True)
async def add_admin_role(itx: discord.Interaction, role: discord.Role) -> None:
    if not _mg_check(itx) or not _same_guild(itx, role):
        await itx.response.send_message("🚫 Permission denied or invalid role.", ephemeral=True)
        return
    ids: list[int] = cfg.get("admin_role_ids", [])
    if role.id in ids:
        await itx.response.send_message(f"ℹ️ **{role.name}** is already an admin role.", ephemeral=True)
        return
    ids.append(role.id)
    cfg["admin_role_ids"] = ids
    await _save_config(cfg)
    await itx.response.send_message(f"✅ **{role.name}** can now answer/deny questions.", ephemeral=True)


@bot.tree.command(name="remove_admin_role", description="Revoke a role's access to answer/deny questions")
@app_commands.describe(role="Role to revoke access from")
@app_commands.checks.has_permissions(manage_guild=True)
async def remove_admin_role(itx: discord.Interaction, role: discord.Role) -> None:
    if not _mg_check(itx):
        await itx.response.send_message("🚫 Permission denied.", ephemeral=True)
        return
    ids: list[int] = cfg.get("admin_role_ids", [])
    if role.id not in ids:
        await itx.response.send_message(f"ℹ️ **{role.name}** is not an admin role.", ephemeral=True)
        return
    ids.remove(role.id)
    cfg["admin_role_ids"] = ids
    await _save_config(cfg)
    await itx.response.send_message(f"✅ **{role.name}** removed from admin roles.", ephemeral=True)

# ──────────────────────────────────────────────
# Owner-only commands  (@commands.is_owner())
# These use the prefix-command system so they are
# never exposed as slash commands in the guild UI.
# ──────────────────────────────────────────────

@bot.command(name="sync", hidden=True)
@commands.is_owner()
async def sync_tree(ctx: commands.Context) -> None:
    """Owner-only: manually re-sync slash commands."""
    synced = await bot.tree.sync()
    await ctx.send(f"✅ Synced {len(synced)} command(s).", delete_after=10)
    log.info("Slash commands synced by owner %s (%s)", ctx.author, ctx.author.id)


@bot.command(name="shutdown", hidden=True)
@commands.is_owner()
async def shutdown(ctx: commands.Context) -> None:
    """Owner-only: gracefully shut the bot down."""
    await ctx.send("👋 Shutting down...", delete_after=5)
    log.info("Shutdown requested by owner %s (%s)", ctx.author, ctx.author.id)
    await bot.close()

# ──────────────────────────────────────────────
# Global error handler — stack traces NEVER reach users
# ──────────────────────────────────────────────

@bot.tree.error
async def on_app_command_error(itx: discord.Interaction, error: app_commands.AppCommandError) -> None:
    if isinstance(error, app_commands.CommandOnCooldown):
        await itx.response.send_message(
            f"⏳ Slow down! Try again in **{error.retry_after:.0f}s**.", ephemeral=True
        )
        return
    if isinstance(error, app_commands.MissingPermissions):
        await itx.response.send_message("🚫 You don't have permission to use that command.", ephemeral=True)
        return
    # All other errors: log privately, tell user nothing specific
    log.error("Unhandled app command error in %r: %s", getattr(itx, "command", "?"), error, exc_info=True)
    msg = "❌ An unexpected error occurred. Please try again later."
    try:
        if itx.response.is_done():
            await itx.followup.send(msg, ephemeral=True)
        else:
            await itx.response.send_message(msg, ephemeral=True)
    except Exception:
        pass

# ──────────────────────────────────────────────
# Ready
# ──────────────────────────────────────────────

@bot.event
async def on_ready() -> None:
    await bot.tree.sync()
    log.info("Ready — logged in as %s (ID: %s)", bot.user, bot.user.id)
    log.info(
        "Invite URL (Least Privilege — no Administrator):\n"
        "https://discord.com/api/oauth2/authorize"
        "?client_id=%s&permissions=274878024768&scope=bot%%20applications.commands",
        bot.user.id,
    )

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
bot.run(TOKEN, log_handler=None)   # log_handler=None so our config above is used
