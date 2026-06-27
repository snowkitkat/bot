// ─────────────────────────────────────────────────────────────
//  Zaibatsu / Neon District Ticket Bot
//  Reports & Appeals  •  JSON panel  •  Threads  •  DM relay
// ─────────────────────────────────────────────────────────────
require('dotenv').config();

const {
  Client, GatewayIntentBits, Partials,
  EmbedBuilder, ButtonBuilder, ButtonStyle, ActionRowBuilder,
  ModalBuilder, TextInputBuilder, TextInputStyle,
  StringSelectMenuBuilder, SlashCommandBuilder,
  REST, Routes, PermissionFlagsBits, MessageFlags, ChannelType,
} = require('discord.js');

const fs   = require('fs');
const path = require('path');

// ─────────────────────────────────────────────────────────────
//  Client
// ─────────────────────────────────────────────────────────────
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.DirectMessages,
  ],
  partials: [Partials.Channel, Partials.Message],
});

// ─────────────────────────────────────────────────────────────
//  Persistence
// ─────────────────────────────────────────────────────────────
const CONFIG_FILE  = path.join(__dirname, 'config.json');
const TICKETS_FILE = path.join(__dirname, 'tickets.json');

function readFile(f) {
  if (!fs.existsSync(f)) { fs.writeFileSync(f, '{}'); return {}; }
  try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch { return {}; }
}
function writeFile(f, d) { fs.writeFileSync(f, JSON.stringify(d, null, 2)); }

function getGuildConfig(guildId) { return readFile(CONFIG_FILE)[guildId] || {}; }
function setGuildConfig(guildId, key, value) {
  const cfg = readFile(CONFIG_FILE);
  if (!cfg[guildId]) cfg[guildId] = {};
  cfg[guildId][key] = value;
  writeFile(CONFIG_FILE, cfg);
}

function getTicket(refId) { return readFile(TICKETS_FILE)[refId] || null; }
function setTicket(refId, data) {
  const t = readFile(TICKETS_FILE);
  t[refId] = data;
  writeFile(TICKETS_FILE, t);
}
function findTicketByDmMsgId(messageId) {
  for (const [refId, ticket] of Object.entries(readFile(TICKETS_FILE)))
    if ((ticket.dmMessageIds || []).includes(messageId)) return { refId, ticket };
  return null;
}

// ─────────────────────────────────────────────────────────────
//  Ticket types
// ─────────────────────────────────────────────────────────────
const TICKET_TYPES = [
  { key: 'report', label: 'Report', prefix: 'RPT', tier: 'support', resolutionStyle: 'handleInsuff', emoji: '🚨', buttonStyle: ButtonStyle.Danger  },
  { key: 'appeal', label: 'Appeal', prefix: 'APL', tier: 'admin',   resolutionStyle: 'acceptDeny',   emoji: '⚖️', buttonStyle: ButtonStyle.Primary },
];
function typeMeta(key) { return TICKET_TYPES.find(t => t.key === key); }

const PLATFORM_OPTIONS = [
  { label: 'Neon District 2051', value: 'Neon District 2051' },
  { label: 'Zaibatsu',           value: 'Zaibatsu'           },
  { label: 'Discord',            value: 'Discord'            },
];

const TERMINAL_STATUSES = ['handled', 'insufficient', 'accepted', 'denied', 'ignored'];
function isTerminal(s) { return TERMINAL_STATUSES.includes(s); }

// ─────────────────────────────────────────────────────────────
//  In-memory session state
//  pendingEvidenceSubmissions: holds form data after the modal
//  is submitted, waiting for the user to provide evidence in DMs.
//  The ticket is NOT created until evidence (or "None") is received.
// ─────────────────────────────────────────────────────────────
const INACTIVITY_MS = 10 * 60 * 1000;
const pendingPlatform            = new Map(); // userId → platform/game string (between dropdown and modal)
const pendingEvidenceSubmissions = new Map(); // userId → { guildId, platform, data, expires }

function freshExpiry()  { return Date.now() + INACTIVITY_MS; }
function isExpired(s)   { return Date.now() > s.expires; }
function touchExpiry(s) { s.expires = freshExpiry(); }

// ─────────────────────────────────────────────────────────────
//  Q&A in-memory state
// ─────────────────────────────────────────────────────────────
const QA_COOLDOWN_MS  = 30 * 1000; // 30 seconds between submissions
const qaLastSubmission = new Map(); // userId → timestamp

function qaCanSubmit(userId) {
  const last = qaLastSubmission.get(userId);
  if (!last) return { ok: true, remaining: 0 };
  const remaining = QA_COOLDOWN_MS - (Date.now() - last);
  return remaining > 0 ? { ok: false, remaining: Math.ceil(remaining / 1000) } : { ok: true, remaining: 0 };
}
function qaRecordSubmission(userId) { qaLastSubmission.set(userId, Date.now()); }

// Sweep stale sessions and notify the user their submission timed out
setInterval(async () => {
  for (const [userId, s] of pendingEvidenceSubmissions) {
    if (Date.now() > s.expires) {
      pendingEvidenceSubmissions.delete(userId);
      await tryDM(userId, {
        embeds: [new EmbedBuilder().setTitle('⏱️  Submission Expired').setColor(0x95A5A6)
          .setDescription('Your ticket was **not submitted** because no evidence was provided within **10 minutes**.\n\nPlease start over from the ticket panel if you\'d still like to submit.')]
      });
    }
  }
}, 30_000);

// ─────────────────────────────────────────────────────────────
//  Utilities
// ─────────────────────────────────────────────────────────────
function genRefId(prefix) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let id = '';
  for (let i = 0; i < 6; i++) id += chars[Math.floor(Math.random() * 36)];
  return `${prefix}-${id}`;
}
function ts() { return `<t:${Math.floor(Date.now() / 1000)}:F>`; }
async function sendDM(userId, opts) {
  try { const u = await client.users.fetch(userId); await u.send(opts); return true; } catch { return false; }
}
async function sendDMGetMsg(userId, opts) {
  try { const u = await client.users.fetch(userId); return await u.send(opts); } catch { return null; }
}
async function tryDM(userId, opts) { await sendDM(userId, opts); }

const E       = MessageFlags.Ephemeral;
const NO_PERM = { content: '❌  You don\'t have permission to use this command.', flags: E };

// ─────────────────────────────────────────────────────────────
//  Permission helpers
// ─────────────────────────────────────────────────────────────
async function resolveFreshMember(interaction) {
  if (!interaction.guild) return interaction.member;
  try { return await interaction.guild.members.fetch(interaction.user.id); } catch { return interaction.member; }
}
function isAdmin(member) { return member?.permissions?.has(PermissionFlagsBits.Administrator) ?? false; }
function canHandle(member, typeKey, guildId) {
  if (!member) return false;
  if (member.permissions?.has(PermissionFlagsBits.Administrator)) return true;
  const cfg = getGuildConfig(guildId);
  if ((cfg.adminRoles || []).some(id => member.roles.cache.has(id))) return true;
  if (typeMeta(typeKey)?.tier === 'support')
    return (cfg.supportRoles || []).some(id => member.roles.cache.has(id));
  return false;
}
function isStaff(member, guildId) {
  if (!member) return false;
  if (member.permissions?.has(PermissionFlagsBits.Administrator)) return true;
  const cfg = getGuildConfig(guildId);
  return [...(cfg.adminRoles || []), ...(cfg.supportRoles || [])].some(id => member.roles.cache.has(id));
}
function fallbackPing(typeKey, guildId) {
  const cfg = getGuildConfig(guildId);
  const ids = typeMeta(typeKey)?.tier === 'admin'
    ? (cfg.adminRoles || [])
    : [...(cfg.adminRoles || []), ...(cfg.supportRoles || [])];
  if (!ids.length) return '*(no staff roles configured)*';
  return [...new Set(ids)].map(id => `<@&${id}>`).join(' ');
}

// ── Q&A config helpers ────────────────────────────────────────
function getQaConfig(guildId) {
  return getGuildConfig(guildId).qa || { targetChannelId: null, answerChannelId: null, banRoleId: null, adminRoleIds: [] };
}
function setQaConfig(guildId, data) { setGuildConfig(guildId, 'qa', data); }

// QA moderator = Manage Guild permission OR a configured QA admin role
function canModerateQa(member, guildId) {
  if (!member) return false;
  if (member.permissions?.has(PermissionFlagsBits.ManageGuild)) return true;
  const qa = getQaConfig(guildId);
  return (qa.adminRoleIds || []).some(id => member.roles.cache.has(id));
}

// ─────────────────────────────────────────────────────────────
//  Slash commands
// ─────────────────────────────────────────────────────────────
const TYPE_CHOICES = TICKET_TYPES.map(t => ({ name: t.label, value: t.key }));

const COMMANDS = [
  new SlashCommandBuilder().setName('addsupportrole').setDescription('Add a Support role (can handle Reports)')
    .addRoleOption(o => o.setName('role').setDescription('Role').setRequired(true))
    .setDefaultMemberPermissions(PermissionFlagsBits.Administrator),
  new SlashCommandBuilder().setName('removesupportrole').setDescription('Remove a Support role')
    .addRoleOption(o => o.setName('role').setDescription('Role').setRequired(true))
    .setDefaultMemberPermissions(PermissionFlagsBits.Administrator),
  new SlashCommandBuilder().setName('addadminrole').setDescription('Add an Admin role (can handle Reports & Appeals)')
    .addRoleOption(o => o.setName('role').setDescription('Role').setRequired(true))
    .setDefaultMemberPermissions(PermissionFlagsBits.Administrator),
  new SlashCommandBuilder().setName('removeadminrole').setDescription('Remove an Admin role')
    .addRoleOption(o => o.setName('role').setDescription('Role').setRequired(true))
    .setDefaultMemberPermissions(PermissionFlagsBits.Administrator),
  new SlashCommandBuilder().setName('setticketchannel').setDescription('Set the destination channel for a ticket type')
    .addStringOption(o => o.setName('type').setDescription('Ticket type').setRequired(true).addChoices(...TYPE_CHOICES))
    .addChannelOption(o => o.setName('channel').setDescription('Channel').setRequired(true).addChannelTypes(ChannelType.GuildText))
    .setDefaultMemberPermissions(PermissionFlagsBits.Administrator),
  new SlashCommandBuilder().setName('postpanel').setDescription('Post the ticket panel using your embed JSON')
    .addChannelOption(o => o.setName('channel').setDescription('Channel to post in').setRequired(true).addChannelTypes(ChannelType.GuildText))
    .addAttachmentOption(o => o.setName('json').setDescription('Embed .json file').setRequired(true))
    .setDefaultMemberPermissions(PermissionFlagsBits.Administrator),
  new SlashCommandBuilder().setName('ticketconfig').setDescription('View ticket channel & role configuration')
    .setDefaultMemberPermissions(PermissionFlagsBits.ManageMessages),
  new SlashCommandBuilder().setName('viewticket').setDescription('Look up a ticket by reference ID')
    .addStringOption(o => o.setName('refid').setDescription('e.g. RPT-ABC123').setRequired(true))
    .setDefaultMemberPermissions(PermissionFlagsBits.ManageMessages),

  // ── Q&A system ───────────────────────────────────────────────
  new SlashCommandBuilder().setName('question').setDescription('Submit a question to the staff team')
    .addBooleanOption(o => o.setName('anonymous').setDescription('Submit anonymously?').setRequired(true)),
  new SlashCommandBuilder().setName('qa-set-target').setDescription('Set the channel where questions are posted')
    .addChannelOption(o => o.setName('channel').setDescription('Channel').setRequired(true).addChannelTypes(ChannelType.GuildText))
    .setDefaultMemberPermissions(PermissionFlagsBits.ManageGuild),
  new SlashCommandBuilder().setName('qa-set-answers').setDescription('Set the channel where answers are posted')
    .addChannelOption(o => o.setName('channel').setDescription('Channel').setRequired(true).addChannelTypes(ChannelType.GuildText))
    .setDefaultMemberPermissions(PermissionFlagsBits.ManageGuild),
  new SlashCommandBuilder().setName('qa-set-ban-role').setDescription('Set a role that cannot submit questions')
    .addRoleOption(o => o.setName('role').setDescription('Role to ban').setRequired(true))
    .setDefaultMemberPermissions(PermissionFlagsBits.ManageGuild),
  new SlashCommandBuilder().setName('qa-add-admin').setDescription('Grant a role the ability to answer/deny questions')
    .addRoleOption(o => o.setName('role').setDescription('Role').setRequired(true))
    .setDefaultMemberPermissions(PermissionFlagsBits.ManageGuild),
  new SlashCommandBuilder().setName('qa-remove-admin').setDescription("Revoke a role's ability to answer/deny questions")
    .addRoleOption(o => o.setName('role').setDescription('Role').setRequired(true))
    .setDefaultMemberPermissions(PermissionFlagsBits.ManageGuild),
].map(c => c.toJSON());

// ─────────────────────────────────────────────────────────────
//  Ready
// ─────────────────────────────────────────────────────────────
client.once('clientReady', async () => {
  console.log(`✅  Bot online as ${client.user.tag}`);
  const rest = new REST({ version: '10' }).setToken(process.env.TOKEN);
  try {
    await rest.put(Routes.applicationCommands(client.user.id), { body: COMMANDS });
    console.log('✅  Slash commands registered globally');
  } catch (err) { console.error('❌  Command registration failed:', err); }
});

// ─────────────────────────────────────────────────────────────
//  DM message listener
// ─────────────────────────────────────────────────────────────
client.on('messageCreate', async (message) => {
  if (message.author.bot) return;
  if (!message.channel.isDMBased()) return;

  // ── 1. Evidence gate — waiting for evidence after report modal ─
  const pending = pendingEvidenceSubmissions.get(message.author.id);
  if (pending) {
    if (isExpired(pending)) {
      pendingEvidenceSubmissions.delete(message.author.id);
      await message.reply({ embeds: [new EmbedBuilder().setTitle('⏱️  Submission Expired').setColor(0x95A5A6)
        .setDescription('This submission window expired. Please start over from the ticket panel.')] });
      return;
    }
    // Any message resets the inactivity timer
    touchExpiry(pending);

    const isNone   = message.content.trim().toLowerCase() === 'none';
    const hasFiles = message.attachments.size > 0;
    const hasLinks = message.content.trim().length > 0 && !isNone;

    // Must provide something — files, links, or explicitly "None"
    if (!isNone && !hasFiles && !hasLinks) {
      await message.reply(
        '⚠️  Please provide your evidence:\n' +
        '• **Attach files** (screenshots, videos)\n' +
        '• **Paste links** (Gyazo, Imgur, CDN links, etc.)\n' +
        '• You can include both files and links in the same message\n' +
        '• Type `None` if you have no evidence'
      );
      return;
    }

    pendingEvidenceSubmissions.delete(message.author.id);

    const fileUrls     = hasFiles ? [...message.attachments.values()].map(a => a.url) : [];
    const evidenceText = hasLinks ? message.content.trim() : null;

    await finalizeReport(message, pending, fileUrls, evidenceText);
    return;
  }

  // ── 2. Explicit reply to a tracked staff DM → relay to thread ─
  const refMsgId = message.reference?.messageId;
  if (refMsgId) {
    const found = findTicketByDmMsgId(refMsgId);
    if (found) { await relayReply(message, found.refId, found.ticket); return; }
  }

  await message.reply({
    embeds: [new EmbedBuilder().setTitle('🤔  Not Sure What This Is For').setColor(0x95A5A6)
      .setDescription('To reply to an existing ticket, use Discord\'s **Reply** feature on the staff message. To open a new ticket, head to the ticket panel in the server.')]
  });
});

// ─────────────────────────────────────────────────────────────
//  Finalize report — called once evidence is received via DM
// ─────────────────────────────────────────────────────────────
async function finalizeReport(triggerMessage, pending, fileUrls, evidenceText) {
  const { guildId, platform, formData, userId } = pending;

  const cfg       = getGuildConfig(guildId);
  const channelId = (cfg.channels || {}).report;
  if (!channelId) { await triggerMessage.reply('❌  Report channel not configured. Contact an administrator.'); return; }
  const channel = await client.channels.fetch(channelId).catch(() => null);
  if (!channel)  { await triggerMessage.reply('❌  Could not find the report channel.'); return; }

  // Build the evidence field value for the embed
  const evidenceParts = [];
  if (evidenceText)      evidenceParts.push(evidenceText);
  if (fileUrls.length)   evidenceParts.push(fileUrls.join('\n'));
  const evidenceValue = evidenceParts.length ? evidenceParts.join('\n') : null;

  const refId = genRefId('RPT');
  const data  = { ...formData, platform, evidence: evidenceValue };

  const ticketData = {
    type: 'report', userId, guildId, data,
    status: 'open', createdAt: Date.now(),
    claimedBy: null, involvedStaff: [], dmMessageIds: [],
    threadId: null, channelId: null, messageId: null,
  };
  setTicket(refId, ticketData);

  const msg = await channel.send({
    embeds: [buildReportEmbed(refId, userId, data)],
    components: buildActionRows(refId, 'report', { claimed: false, terminal: false }),
  });

  ticketData.channelId = channel.id;
  ticketData.messageId = msg.id;
  ticketData.threadId  = await createThread(msg, refId);
  setTicket(refId, ticketData);

  // Forward any attached files into the thread
  if (fileUrls.length && ticketData.threadId) {
    const thread = await client.channels.fetch(ticketData.threadId).catch(() => null);
    if (thread) await thread.send({ content: `📎  **Evidence files** for \`${refId}\`:`, files: fileUrls }).catch(() => {});
  }

  // DM confirmation
  const confirmMsg = await sendDMGetMsg(userId, {
    embeds: [new EmbedBuilder().setTitle('✅  Report Submitted').setColor(0x57F287)
      .setDescription(`Your report has been submitted to the moderation team.\n\n**Reference ID:** \`${refId}\`\n\nKeep this ID — all updates will reference it.\n\nHave more to add? Reply to this DM at any time.`)
      .setTimestamp()]
  });
  if (confirmMsg) { ticketData.dmMessageIds.push(confirmMsg.id); setTicket(refId, ticketData); }
}

// ─────────────────────────────────────────────────────────────
//  Relay reporter DM reply → ticket thread
// ─────────────────────────────────────────────────────────────
async function relayReply(message, refId, ticket) {
  if (isTerminal(ticket.status)) {
    await message.reply({ embeds: [new EmbedBuilder().setTitle('🔒  Ticket Closed').setColor(0x95A5A6)
      .setDescription(`Your ticket **\`${refId}\`** is already **${ticket.status}** and closed. This message was **not** sent to staff.\n\nOpen a new ticket if you need further help.`)
      .setTimestamp()] });
    return;
  }
  const thread = ticket.threadId ? await client.channels.fetch(ticket.threadId).catch(() => null) : null;
  if (!thread) { await message.reply('❌  Could not reach the ticket thread. Please contact staff directly.'); return; }

  const mentions = ticket.involvedStaff?.length
    ? ticket.involvedStaff.map(id => `<@${id}>`).join(' ')
    : fallbackPing(ticket.type, ticket.guildId);

  await thread.send({
    content: `${mentions}\n📩  **Reporter replied** (\`${refId}\`):`,
    embeds:  [new EmbedBuilder().setColor(0x5865F2).setDescription(message.content || '*(see attachments)*').setFooter({ text: message.author.tag }).setTimestamp()],
    files:   [...message.attachments.values()].map(a => a.url),
  });
  await message.reply({ embeds: [new EmbedBuilder().setTitle('✅  Message Forwarded').setColor(0x57F287)
    .setDescription(`Your reply for **\`${refId}\`** has been sent to staff.`).setTimestamp()] });
}

// ─────────────────────────────────────────────────────────────
//  Ticket embeds
// ─────────────────────────────────────────────────────────────
function buildReportEmbed(refId, userId, d) {
  return new EmbedBuilder().setTitle(`🚨  Report Ticket — \`${refId}\``).setColor(0x5865F2)
    .addFields(
      { name: '🔖 Reference ID',  value: `\`${refId}\``, inline: true },
      { name: '📨 Submitted By',  value: `<@${userId}>`, inline: true },
      { name: '📅 Submitted At',  value: ts(),           inline: true },
      { name: '\u200B',           value: '\u200B',        inline: false },
      { name: '🎮 Platform',      value: d.platform,     inline: true  },
      { name: '📛 Category',      value: d.category,     inline: true  },
      { name: '\u200B',           value: '\u200B',        inline: true  },
      { name: '👤 Offender Username',     value: `\`${d.offenderUsername}\``,    inline: true },
      { name: '🪪  Offender UserID',      value: `\`${d.offenderUserId}\``,      inline: true },
      { name: '💬 Offender Display Name', value: `\`${d.offenderDisplayName}\``, inline: true },
      { name: '📝 Description',   value: d.description },
      { name: '🔗 Evidence',      value: d.evidence || '*No evidence provided*' },
    )
    .setFooter({ text: '⚪  Status: Open' });
}

function buildAppealEmbed(refId, userId, d) {
  return new EmbedBuilder().setTitle(`⚖️  Appeal Ticket — \`${refId}\``).setColor(0x5865F2)
    .addFields(
      { name: '🔖 Reference ID',  value: `\`${refId}\``, inline: true },
      { name: '📨 Submitted By',  value: `<@${userId}>`, inline: true },
      { name: '📅 Submitted At',  value: ts(),           inline: true },
      { name: '\u200B',           value: '\u200B',        inline: false },
      { name: '🎮 Game',        value: d.game,        inline: true },
      { name: '⚖️  Punishment', value: d.punishment,  inline: true },
      { name: '\u200B',         value: '\u200B',       inline: true },
      { name: '👤 Username',    value: `\`${d.username}\``, inline: true },
      { name: '🪪  User ID',    value: `\`${d.userId}\``,   inline: true },
      { name: '\u200B',         value: '\u200B',             inline: true },
      { name: '📝 Reason for Appeal',  value: d.reason },
      { name: 'ℹ️  Additional Notes',  value: d.additional || '*None*' },
    )
    .setFooter({ text: '⚪  Status: Open' });
}

// ─────────────────────────────────────────────────────────────
//  Component builders
// ─────────────────────────────────────────────────────────────
function panelRow() {
  return new ActionRowBuilder().addComponents(
    ...TICKET_TYPES.map(t => new ButtonBuilder().setCustomId(`panel_${t.key}`).setLabel(t.label).setEmoji(t.emoji).setStyle(t.buttonStyle))
  );
}
function platformSelectRow(customId, placeholder) {
  return new ActionRowBuilder().addComponents(
    new StringSelectMenuBuilder().setCustomId(customId).setPlaceholder(placeholder).addOptions(PLATFORM_OPTIONS)
  );
}
function buildActionRows(refId, typeKey, { claimed, terminal }) {
  const row1 = new ActionRowBuilder().addComponents(
    new ButtonBuilder().setCustomId(`claim_${refId}`).setLabel(claimed ? 'Claimed' : '🔒  Claim').setStyle(claimed ? ButtonStyle.Secondary : ButtonStyle.Primary).setDisabled(claimed || terminal),
    new ButtonBuilder().setCustomId(`msg_${refId}`).setLabel('📨  Message Reporter').setStyle(ButtonStyle.Secondary).setDisabled(terminal),
  );
  const row2 = typeMeta(typeKey)?.resolutionStyle === 'handleInsuff'
    ? new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId(`handle_${refId}`).setLabel('Handle').setEmoji('✅').setStyle(ButtonStyle.Success).setDisabled(terminal),
        new ButtonBuilder().setCustomId(`insuff_${refId}`).setLabel('Insufficient Evidence').setEmoji('⚠️').setStyle(ButtonStyle.Primary).setDisabled(terminal),
        new ButtonBuilder().setCustomId(`ignore_${refId}`).setLabel('Ignore').setEmoji('🚫').setStyle(ButtonStyle.Secondary).setDisabled(terminal),
      )
    : new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId(`accept_${refId}`).setLabel('Accept').setEmoji('✅').setStyle(ButtonStyle.Success).setDisabled(terminal),
        new ButtonBuilder().setCustomId(`deny_${refId}`).setLabel('Deny').setEmoji('❌').setStyle(ButtonStyle.Danger).setDisabled(terminal),
        new ButtonBuilder().setCustomId(`ignore_${refId}`).setLabel('Ignore').setEmoji('🚫').setStyle(ButtonStyle.Secondary).setDisabled(terminal),
      );
  return [row1, row2];
}
async function createThread(message, refId) {
  try { return (await message.startThread({ name: refId, autoArchiveDuration: 10080, reason: 'Ticket thread' })).id; }
  catch (err) { console.error(`Thread creation failed for ${refId}:`, err.message); return null; }
}
function normalizeEmbedJson(parsed) {
  let content, embedsArr;
  if (Array.isArray(parsed)) { embedsArr = parsed; }
  else if (parsed.embeds)    { embedsArr = parsed.embeds; content = parsed.content; }
  else                       { embedsArr = [parsed]; }
  return { content, embeds: embedsArr.slice(0, 10).map(e => EmbedBuilder.from(e)) };
}

// ─────────────────────────────────────────────────────────────
//  Interaction router
// ─────────────────────────────────────────────────────────────
client.on('interactionCreate', async (interaction) => {
  try {
    if      (interaction.isChatInputCommand())  await handleCommand(interaction);
    else if (interaction.isStringSelectMenu())  await handleSelect(interaction);
    else if (interaction.isButton())            await handleButton(interaction);
    else if (interaction.isModalSubmit())       await handleModal(interaction);
  } catch (err) {
    console.error('Interaction error:', err);
    try {
      if (!interaction.replied && !interaction.deferred)
        await interaction.reply({ content: '❌  An unexpected error occurred.', flags: E });
    } catch { /* ignore */ }
  }
});

// ─────────────────────────────────────────────────────────────
//  Slash command handler
// ─────────────────────────────────────────────────────────────
async function handleCommand(interaction) {
  const { commandName, guildId } = interaction;
  const member = await resolveFreshMember(interaction);

  if (['addsupportrole','removesupportrole','addadminrole','removeadminrole'].includes(commandName)) {
    if (!isAdmin(member)) return interaction.reply(NO_PERM);
    const role  = interaction.options.getRole('role');
    const key   = commandName.includes('support') ? 'supportRoles' : 'adminRoles';
    const label = commandName.includes('support') ? 'Support role' : 'Admin role';
    const cfg   = getGuildConfig(guildId);
    const list  = cfg[key] || [];
    if (commandName.startsWith('add')) {
      if (list.includes(role.id)) return interaction.reply({ content: `⚠️  ${role} is already a ${label}.`, flags: E });
      list.push(role.id); setGuildConfig(guildId, key, list);
      return interaction.reply({ content: `✅  ${role} added as a ${label}.`, flags: E });
    } else {
      const idx = list.indexOf(role.id);
      if (idx === -1) return interaction.reply({ content: `⚠️  ${role} is not a configured ${label}.`, flags: E });
      list.splice(idx, 1); setGuildConfig(guildId, key, list);
      return interaction.reply({ content: `✅  ${role} removed from ${label}s.`, flags: E });
    }
  }

  if (commandName === 'setticketchannel') {
    if (!isAdmin(member)) return interaction.reply(NO_PERM);
    const typeKey  = interaction.options.getString('type');
    const ch       = interaction.options.getChannel('channel');
    const cfg      = getGuildConfig(guildId);
    const channels = cfg.channels || {};
    channels[typeKey] = ch.id; setGuildConfig(guildId, 'channels', channels);
    return interaction.reply({ content: `✅  ${typeMeta(typeKey).label} tickets → ${ch}.`, flags: E });
  }

  if (commandName === 'postpanel') {
    if (!isAdmin(member)) return interaction.reply(NO_PERM);
    const ch  = interaction.options.getChannel('channel');
    const att = interaction.options.getAttachment('json');
    await interaction.deferReply({ flags: E });
    let text;
    try { const res = await fetch(att.url); text = await res.text(); }
    catch { return interaction.editReply({ content: '❌  Could not download the JSON file.' }); }
    let parsed;
    try { parsed = JSON.parse(text); }
    catch { return interaction.editReply({ content: '❌  That file is not valid JSON.' }); }
    let norm;
    try { norm = normalizeEmbedJson(parsed); }
    catch { return interaction.editReply({ content: '❌  Could not parse as a Discord embed.' }); }
    const target = await client.channels.fetch(ch.id).catch(() => null);
    if (!target) return interaction.editReply({ content: '❌  Could not access that channel.' });
    await target.send({ content: norm.content || undefined, embeds: norm.embeds, components: [panelRow()] });
    return interaction.editReply({ content: `✅  Ticket panel posted in ${ch}.` });
  }

  if (commandName === 'ticketconfig') {
    if (!isStaff(member, guildId)) return interaction.reply(NO_PERM);
    const cfg      = getGuildConfig(guildId);
    const channels = cfg.channels || {};
    return interaction.reply({
      embeds: [new EmbedBuilder().setTitle('⚙️  Ticket Configuration').setColor(0x5865F2).addFields(
        { name: '🚨  Report Channel',              value: channels.report ? `<#${channels.report}>` : '`Not set`' },
        { name: '⚖️  Appeal Channel',              value: channels.appeal ? `<#${channels.appeal}>` : '`Not set`' },
        { name: '🛡️  Support Roles (Reports)',     value: (cfg.supportRoles || []).map(id => `<@&${id}>`).join(', ') || '*None*' },
        { name: '👑  Admin Roles (All Tickets)',   value: (cfg.adminRoles   || []).map(id => `<@&${id}>`).join(', ') || '*None*' },
      )],
      flags: E,
    });
  }

  if (commandName === 'viewticket') {
    if (!isStaff(member, guildId)) return interaction.reply(NO_PERM);
    const refId  = interaction.options.getString('refid').trim().toUpperCase();
    const ticket = getTicket(refId);
    if (!ticket) return interaction.reply({ content: `❌  No ticket found with ID \`${refId}\`.`, flags: E });
    if (!canHandle(member, ticket.type, guildId))
      return interaction.reply({ content: `❌  You don't have permission to view **${typeMeta(ticket.type)?.label}** tickets.`, flags: E });
    const statusLabels = { open:'⚪ Open', claimed:'🔒 Claimed', handled:'✅ Handled', insufficient:'⚠️ Insufficient Evidence', accepted:'✅ Accepted', denied:'❌ Denied', ignored:'🚫 Ignored' };
    const embed = new EmbedBuilder().setTitle(`🔍  Ticket — \`${refId}\``).setColor(0x5865F2).addFields(
      { name: 'Type',      value: typeMeta(ticket.type)?.label || ticket.type,    inline: true },
      { name: 'Status',    value: statusLabels[ticket.status] || ticket.status,   inline: true },
      { name: 'Submitted', value: `<t:${Math.floor(ticket.createdAt / 1000)}:F>`, inline: true },
      { name: 'Reporter',  value: `<@${ticket.userId}>`,                          inline: true },
    );
    if (ticket.claimedBy) embed.addFields({ name: 'Claimed By',     value: `<@${ticket.claimedBy}>`, inline: true });
    if (ticket.modNote)   embed.addFields({ name: 'Moderator Note', value: ticket.modNote });
    return interaction.reply({ embeds: [embed], flags: E });
  }

  // ── Q&A setup commands ────────────────────────────────────────
  if (commandName === 'question') {
    if (!interaction.guild) return interaction.reply({ content: '❌  Server only.', flags: E });
    const freshMember = await resolveFreshMember(interaction);
    const qa = getQaConfig(guildId);
    if (qa.banRoleId && freshMember.roles.cache.has(qa.banRoleId))
      return interaction.reply({ content: '🚫  You are not allowed to submit questions.', flags: E });
    const cooldown = qaCanSubmit(interaction.user.id);
    if (!cooldown.ok)
      return interaction.reply({ content: `⏳  Please wait **${cooldown.remaining}s** before submitting another question.`, flags: E });
    const anonymous = interaction.options.getBoolean('anonymous');
    const modal = new ModalBuilder().setCustomId(`qa_question_${anonymous ? '1' : '0'}`).setTitle('Submit a Question');
    modal.addComponents(new ActionRowBuilder().addComponents(
      new TextInputBuilder().setCustomId('question').setLabel('Your Question')
        .setStyle(TextInputStyle.Paragraph).setRequired(true).setMaxLength(1000)
    ));
    return interaction.showModal(modal);
  }

  if (commandName === 'qa-set-target') {
    if (!member.permissions.has(PermissionFlagsBits.ManageGuild)) return interaction.reply(NO_PERM);
    const ch = interaction.options.getChannel('channel');
    const qa = getQaConfig(guildId); qa.targetChannelId = ch.id; setQaConfig(guildId, qa);
    return interaction.reply({ content: `✅  Questions will be posted in ${ch}.`, flags: E });
  }
  if (commandName === 'qa-set-answers') {
    if (!member.permissions.has(PermissionFlagsBits.ManageGuild)) return interaction.reply(NO_PERM);
    const ch = interaction.options.getChannel('channel');
    const qa = getQaConfig(guildId); qa.answerChannelId = ch.id; setQaConfig(guildId, qa);
    return interaction.reply({ content: `✅  Answers will be posted in ${ch}.`, flags: E });
  }
  if (commandName === 'qa-set-ban-role') {
    if (!member.permissions.has(PermissionFlagsBits.ManageGuild)) return interaction.reply(NO_PERM);
    const role = interaction.options.getRole('role');
    const qa   = getQaConfig(guildId); qa.banRoleId = role.id; setQaConfig(guildId, qa);
    return interaction.reply({ content: `✅  ${role} cannot submit questions.`, flags: E });
  }
  if (commandName === 'qa-add-admin') {
    if (!member.permissions.has(PermissionFlagsBits.ManageGuild)) return interaction.reply(NO_PERM);
    const role = interaction.options.getRole('role');
    const qa   = getQaConfig(guildId);
    qa.adminRoleIds = qa.adminRoleIds || [];
    if (qa.adminRoleIds.includes(role.id)) return interaction.reply({ content: `⚠️  ${role} is already a Q&A admin.`, flags: E });
    qa.adminRoleIds.push(role.id); setQaConfig(guildId, qa);
    return interaction.reply({ content: `✅  ${role} can now answer and deny questions.`, flags: E });
  }
  if (commandName === 'qa-remove-admin') {
    if (!member.permissions.has(PermissionFlagsBits.ManageGuild)) return interaction.reply(NO_PERM);
    const role = interaction.options.getRole('role');
    const qa   = getQaConfig(guildId);
    qa.adminRoleIds = (qa.adminRoleIds || []).filter(id => id !== role.id);
    setQaConfig(guildId, qa);
    return interaction.reply({ content: `✅  ${role} can no longer answer or deny questions.`, flags: E });
  }
}

// ─────────────────────────────────────────────────────────────
//  Select menu handler
// ─────────────────────────────────────────────────────────────
async function handleSelect(interaction) {
  const { customId, user } = interaction;

  // Report: platform chosen → single 5-field modal
  if (customId === 'sel_report_platform') {
    pendingPlatform.set(user.id, interaction.values[0]);
    const modal = new ModalBuilder().setCustomId('modal_report').setTitle('Report Ticket');
    modal.addComponents(
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('category').setLabel('Category').setPlaceholder('e.g. RDM, NLR, NITRP, FailRP, etc.').setStyle(TextInputStyle.Short).setRequired(true).setMaxLength(100)),
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('offender_username').setLabel("Offender's Username").setStyle(TextInputStyle.Short).setRequired(true).setMaxLength(100)),
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('offender_userid').setLabel("Offender's UserID").setStyle(TextInputStyle.Short).setRequired(true).setMaxLength(100)),
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('offender_displayname').setLabel("Offender's Display Name").setStyle(TextInputStyle.Short).setRequired(true).setMaxLength(100)),
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('description').setLabel('Description').setPlaceholder('Describe what happened in detail...').setStyle(TextInputStyle.Paragraph).setRequired(true).setMaxLength(1000)),
    );
    return interaction.showModal(modal);
  }

  // Appeal: game chosen → single 5-field modal
  if (customId === 'sel_appeal_game') {
    pendingPlatform.set(user.id, interaction.values[0]);
    const modal = new ModalBuilder().setCustomId('modal_appeal').setTitle('Appeal Ticket');
    modal.addComponents(
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('punishment').setLabel('Punishment Type').setPlaceholder('e.g. Ban, Mute, Warning, etc.').setStyle(TextInputStyle.Short).setRequired(true).setMaxLength(100)),
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('username').setLabel('Your Username').setStyle(TextInputStyle.Short).setRequired(true).setMaxLength(100)),
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('user_id').setLabel('Your User ID').setStyle(TextInputStyle.Short).setRequired(true).setMaxLength(100)),
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('reason').setLabel('Reason for Appeal').setStyle(TextInputStyle.Paragraph).setRequired(true).setMaxLength(1000)),
      new ActionRowBuilder().addComponents(new TextInputBuilder().setCustomId('additional').setLabel('Additional Notes (Optional)').setStyle(TextInputStyle.Paragraph).setRequired(false).setMaxLength(500)),
    );
    return interaction.showModal(modal);
  }
}

// ─────────────────────────────────────────────────────────────
//  Button handler
// ─────────────────────────────────────────────────────────────
async function handleButton(interaction) {
  const { customId, guildId, user } = interaction;
  const member = await resolveFreshMember(interaction);

  if (customId === 'panel_report') {
    return interaction.reply({
      embeds: [new EmbedBuilder().setTitle('🚨  Report — Select Platform').setColor(0x5865F2).setDescription('Select the platform where the violation occurred.')],
      components: [platformSelectRow('sel_report_platform', 'Select a platform...')],
      flags: E,
    });
  }
  if (customId === 'panel_appeal') {
    return interaction.reply({
      embeds: [new EmbedBuilder().setTitle('⚖️  Appeal — Select Game').setColor(0x5865F2).setDescription('Select the game you are appealing for.')],
      components: [platformSelectRow('sel_appeal_game', 'Select a game...')],
      flags: E,
    });
  }

  // Claim
  if (customId.startsWith('claim_')) {
    const refId  = customId.slice('claim_'.length);
    const ticket = getTicket(refId);
    if (!ticket)                   return interaction.reply({ content: '❌  Ticket not found.', flags: E });
    if (isTerminal(ticket.status)) return interaction.reply({ content: '⚠️  This ticket has already been finalized.', flags: E });
    if (!canHandle(member, ticket.type, guildId))
      return interaction.reply({ content: `❌  You don't have permission to handle **${typeMeta(ticket.type)?.label}** tickets.`, flags: E });
    if (ticket.claimedBy) return interaction.reply({ content: `⚠️  Already claimed by <@${ticket.claimedBy}>.`, flags: E });

    ticket.claimedBy = user.id; ticket.claimedAt = Date.now(); ticket.status = 'claimed';
    ticket.involvedStaff = ticket.involvedStaff || [];
    if (!ticket.involvedStaff.includes(user.id)) ticket.involvedStaff.push(user.id);
    setTicket(refId, ticket);
    const updated = EmbedBuilder.from(interaction.message.embeds[0]).setColor(0xFEE75C).setFooter({ text: '🔒  Status: Claimed' }).addFields({ name: '🔒  Claimed By', value: `${user} • ${ts()}` });
    await interaction.update({ embeds: [updated], components: buildActionRows(refId, ticket.type, { claimed: true, terminal: false }) });
    await tryDM(ticket.userId, { embeds: [new EmbedBuilder().setTitle('📬  Ticket Update').setColor(0xFEE75C)
      .setDescription(`Your ticket **\`${refId}\`** has been claimed by a member of our staff team.\n\nExpect a response shortly.`).setTimestamp()] });
    return;
  }

  // Message Reporter
  if (customId.startsWith('msg_')) {
    const refId  = customId.slice('msg_'.length);
    const ticket = getTicket(refId);
    if (!ticket)                   return interaction.reply({ content: '❌  Ticket not found.', flags: E });
    if (isTerminal(ticket.status)) return interaction.reply({ content: '⚠️  This ticket has already been finalized.', flags: E });
    if (!canHandle(member, ticket.type, guildId))
      return interaction.reply({ content: `❌  You don't have permission to handle **${typeMeta(ticket.type)?.label}** tickets.`, flags: E });
    const modal = new ModalBuilder().setCustomId(`staffmsg_${refId}`).setTitle('Message Reporter');
    modal.addComponents(new ActionRowBuilder().addComponents(
      new TextInputBuilder().setCustomId('message').setLabel('Message to send to the reporter').setStyle(TextInputStyle.Paragraph).setRequired(true).setMaxLength(1500)
    ));
    return interaction.showModal(modal);
  }

  // Handle → note modal
  if (customId.startsWith('handle_')) {
    const refId  = customId.slice('handle_'.length);
    const ticket = getTicket(refId);
    if (!ticket)                   return interaction.reply({ content: '❌  Ticket not found.', flags: E });
    if (isTerminal(ticket.status)) return interaction.reply({ content: '⚠️  Already finalized.', flags: E });
    if (!canHandle(member, ticket.type, guildId))
      return interaction.reply({ content: '❌  You don\'t have permission to handle this ticket.', flags: E });
    const modal = new ModalBuilder().setCustomId(`handle_note_${refId}`).setTitle('Handle Report — Moderator Note');
    modal.addComponents(new ActionRowBuilder().addComponents(
      new TextInputBuilder().setCustomId('mod_note').setLabel('Moderator Note').setStyle(TextInputStyle.Paragraph).setRequired(true).setMaxLength(500)
    ));
    return interaction.showModal(modal);
  }

  // Insufficient Evidence → note modal
  if (customId.startsWith('insuff_')) {
    const refId  = customId.slice('insuff_'.length);
    const ticket = getTicket(refId);
    if (!ticket)                   return interaction.reply({ content: '❌  Ticket not found.', flags: E });
    if (isTerminal(ticket.status)) return interaction.reply({ content: '⚠️  Already finalized.', flags: E });
    if (!canHandle(member, ticket.type, guildId))
      return interaction.reply({ content: '❌  You don\'t have permission to handle this ticket.', flags: E });
    const modal = new ModalBuilder().setCustomId(`insuff_note_${refId}`).setTitle('Insufficient Evidence — Note');
    modal.addComponents(new ActionRowBuilder().addComponents(
      new TextInputBuilder().setCustomId('mod_note').setLabel('Moderator Note').setStyle(TextInputStyle.Paragraph).setRequired(true).setMaxLength(500)
    ));
    return interaction.showModal(modal);
  }

  // Accept → note modal
  if (customId.startsWith('accept_')) {
    const refId  = customId.slice('accept_'.length);
    const ticket = getTicket(refId);
    if (!ticket)                   return interaction.reply({ content: '❌  Ticket not found.', flags: E });
    if (isTerminal(ticket.status)) return interaction.reply({ content: '⚠️  Already finalized.', flags: E });
    if (!canHandle(member, ticket.type, guildId))
      return interaction.reply({ content: '❌  You don\'t have permission to handle this ticket.', flags: E });
    const modal = new ModalBuilder().setCustomId(`accept_note_${refId}`).setTitle('Accept Appeal — Moderator Note');
    modal.addComponents(new ActionRowBuilder().addComponents(
      new TextInputBuilder().setCustomId('mod_note').setLabel('Moderator Note').setStyle(TextInputStyle.Paragraph).setRequired(true).setMaxLength(500)
    ));
    return interaction.showModal(modal);
  }

  // Deny → note modal
  if (customId.startsWith('deny_')) {
    const refId  = customId.slice('deny_'.length);
    const ticket = getTicket(refId);
    if (!ticket)                   return interaction.reply({ content: '❌  Ticket not found.', flags: E });
    if (isTerminal(ticket.status)) return interaction.reply({ content: '⚠️  Already finalized.', flags: E });
    if (!canHandle(member, ticket.type, guildId))
      return interaction.reply({ content: '❌  You don\'t have permission to handle this ticket.', flags: E });
    const modal = new ModalBuilder().setCustomId(`deny_note_${refId}`).setTitle('Deny Appeal — Moderator Note');
    modal.addComponents(new ActionRowBuilder().addComponents(
      new TextInputBuilder().setCustomId('mod_note').setLabel('Moderator Note').setStyle(TextInputStyle.Paragraph).setRequired(true).setMaxLength(500)
    ));
    return interaction.showModal(modal);
  }

  // Ignore — silent, no DM, no note
  if (customId.startsWith('ignore_')) {
    const refId  = customId.slice('ignore_'.length);
    const ticket = getTicket(refId);
    if (!ticket)                   return interaction.reply({ content: '❌  Ticket not found.', flags: E });
    if (isTerminal(ticket.status)) return interaction.reply({ content: '⚠️  Already finalized.', flags: E });
    if (!canHandle(member, ticket.type, guildId))
      return interaction.reply({ content: '❌  You don\'t have permission to handle this ticket.', flags: E });
    ticket.status = 'ignored'; ticket.handledBy = user.id; ticket.handledAt = Date.now();
    setTicket(refId, ticket);
    const updated = EmbedBuilder.from(interaction.message.embeds[0]).setColor(0x95A5A6).setFooter({ text: '🚫  Status: Ignored' }).addFields({ name: '🚫  Ignored By', value: `${user} • ${ts()}` });
    await interaction.update({ embeds: [updated], components: buildActionRows(refId, ticket.type, { claimed: !!ticket.claimedBy, terminal: true }) });
    if (ticket.threadId) {
      const thread = await client.channels.fetch(ticket.threadId).catch(() => null);
      if (thread) await thread.setArchived(true).catch(() => {});
    }
    return;
  }

  // ── Q&A: Answer button ────────────────────────────────────────
  if (customId.startsWith('qa_answer_')) {
    const freshMember = await resolveFreshMember(interaction);
    if (!canModerateQa(freshMember, guildId))
      return interaction.reply({ content: '🚫  You don\'t have permission to answer questions.', flags: E });
    if (!interaction.message?.embeds?.length)
      return interaction.reply({ content: '❌  Could not read question data.', flags: E });
    const submitterId = customId.slice('qa_answer_'.length);
    const modal = new ModalBuilder().setCustomId(`qa_answer_modal_${submitterId}`).setTitle('Answer this Question');
    modal.addComponents(new ActionRowBuilder().addComponents(
      new TextInputBuilder().setCustomId('answer').setLabel('Your Answer')
        .setStyle(TextInputStyle.Paragraph).setRequired(true).setMaxLength(1000)
    ));
    return interaction.showModal(modal);
  }

  // ── Q&A: Deny button ─────────────────────────────────────────
  if (customId.startsWith('qa_deny_')) {
    const freshMember = await resolveFreshMember(interaction);
    if (!canModerateQa(freshMember, guildId))
      return interaction.reply({ content: '🚫  You don\'t have permission to deny questions.', flags: E });
    if (!interaction.message?.embeds?.length)
      return interaction.reply({ content: '❌  Could not read question data.', flags: E });
    const submitterId = customId.slice('qa_deny_'.length);
    const modal = new ModalBuilder().setCustomId(`qa_deny_modal_${submitterId}`).setTitle('Deny this Question');
    modal.addComponents(new ActionRowBuilder().addComponents(
      new TextInputBuilder().setCustomId('reason').setLabel('Reason (optional)')
        .setStyle(TextInputStyle.Paragraph).setRequired(false).setMaxLength(500)
        .setPlaceholder('Leave blank for no reason provided.')
    ));
    return interaction.showModal(modal);
  }
}

// ─────────────────────────────────────────────────────────────
//  Shared resolution finalize (Handle/Insufficient/Accept/Deny)
// ─────────────────────────────────────────────────────────────
const RES_COLOR  = { handled: 0x57F287, insufficient: 0xFEE75C, accepted: 0x57F287, denied: 0xED4245 };
const RES_FOOTER = { handled: '✅  Status: Handled', insufficient: '⚠️  Status: Insufficient Evidence', accepted: '✅  Status: Accepted', denied: '❌  Status: Denied' };
const RES_DM = {
  handled:      (refId, note)         => `Your report **\`${refId}\`** has been reviewed.\n\nThank you for your report. The reported user(s) have been moderated.\n\n**Moderator Note:** ${note}`,
  insufficient: (refId, note)         => `Your report **\`${refId}\`** has been reviewed.\n\nWe were unable to properly process your report due to a lack of evidence.\n\n**Moderator Note:** ${note}`,
  accepted:     (refId, note, ticket) => `Your **${typeMeta(ticket.type)?.label}** (**\`${refId}\`**) has been **accepted**.\n\n**Moderator Note:** ${note}`,
  denied:       (refId, note, ticket) => `Your **${typeMeta(ticket.type)?.label}** (**\`${refId}\`**) has been **denied**.\n\n**Moderator Note:** ${note}`,
};

async function finalizeTicket(interaction, refId, statusKey) {
  const ticket = getTicket(refId);
  if (!ticket)                   return interaction.reply({ content: '❌  Ticket not found.', flags: E });
  if (isTerminal(ticket.status)) return interaction.reply({ content: '⚠️  Already finalized.', flags: E });

  const note = interaction.fields.getTextInputValue('mod_note');
  ticket.status = statusKey; ticket.handledBy = interaction.user.id; ticket.handledAt = Date.now(); ticket.modNote = note;
  setTicket(refId, ticket);

  const ch  = await client.channels.fetch(ticket.channelId).catch(() => null);
  const msg = ch ? await ch.messages.fetch(ticket.messageId).catch(() => null) : null;
  if (msg) {
    await msg.edit({
      embeds: [EmbedBuilder.from(msg.embeds[0]).setColor(RES_COLOR[statusKey]).setFooter({ text: RES_FOOTER[statusKey] })
        .addFields({ name: '🛡️  Reviewed By', value: `${interaction.user} • ${ts()}` }, { name: '📝  Moderator Note', value: note })
      ],
      components: buildActionRows(refId, ticket.type, { claimed: !!ticket.claimedBy, terminal: true }),
    });
  }
  if (ticket.threadId) {
    const thread = await client.channels.fetch(ticket.threadId).catch(() => null);
    if (thread) await thread.setArchived(true).catch(() => {});
  }
  await interaction.reply({ content: `✅  Ticket marked as **${statusKey}**.`, flags: E });
  await tryDM(ticket.userId, { embeds: [new EmbedBuilder().setTitle('📬  Ticket Update').setColor(RES_COLOR[statusKey])
    .setDescription(RES_DM[statusKey](refId, note, ticket)).setTimestamp()] });
}

// ─────────────────────────────────────────────────────────────
//  Modal handler
// ─────────────────────────────────────────────────────────────
async function handleModal(interaction) {
  const { customId, guildId, user } = interaction;

  // ── Report modal — store form data, DM for evidence ──────────
  if (customId === 'modal_report') {
    const platform = pendingPlatform.get(user.id) || 'Unknown';
    pendingPlatform.delete(user.id);

    const formData = {
      category:            interaction.fields.getTextInputValue('category'),
      offenderUsername:    interaction.fields.getTextInputValue('offender_username'),
      offenderUserId:      interaction.fields.getTextInputValue('offender_userid'),
      offenderDisplayName: interaction.fields.getTextInputValue('offender_displayname'),
      description:         interaction.fields.getTextInputValue('description'),
    };

    // Store the submission — ticket won't be created until evidence is received
    pendingEvidenceSubmissions.set(user.id, {
      guildId, platform, formData, userId: user.id, expires: freshExpiry(),
    });

    const dmSent = await sendDM(user.id, {
      embeds: [new EmbedBuilder().setTitle('📎  Evidence Required').setColor(0x5865F2)
        .setDescription(
          'Almost done! Please provide evidence for your report within **10 minutes**.\n\n' +
          '• **Attach files** — screenshots, screen recordings, etc.\n' +
          '• **Paste links** — Gyazo, Imgur, CDN links, etc.\n' +
          '• You can include **both files and links** in the same message.\n' +
          '• Type `None` if you have no evidence to provide.\n\n' +
          '⏱️  **Your report will not be submitted if you don\'t respond in time.**'
        )
        .setTimestamp()]
    });

    if (!dmSent) {
      pendingEvidenceSubmissions.delete(user.id);
      return interaction.reply({ content: '❌  Could not DM you. Please enable **Allow direct messages from server members** in your Privacy Settings and try again.', flags: E });
    }
    return interaction.reply({ content: '📬  Check your DMs! You have **10 minutes** to provide evidence to complete your report.', flags: E });
  }

  // ── Appeal modal — create ticket immediately ──────────────────
  if (customId === 'modal_appeal') {
    const game = pendingPlatform.get(user.id) || 'Unknown';
    pendingPlatform.delete(user.id);

    const cfg       = getGuildConfig(guildId);
    const channelId = (cfg.channels || {}).appeal;
    if (!channelId) return interaction.reply({ content: '❌  Appeal channel not configured. Contact an administrator.', flags: E });
    const channel   = await client.channels.fetch(channelId).catch(() => null);
    if (!channel)   return interaction.reply({ content: '❌  Could not find the appeal channel.', flags: E });

    const refId = genRefId('APL');
    const data  = {
      game,
      punishment: interaction.fields.getTextInputValue('punishment'),
      username:   interaction.fields.getTextInputValue('username'),
      userId:     interaction.fields.getTextInputValue('user_id'),
      reason:     interaction.fields.getTextInputValue('reason'),
      additional: interaction.fields.getTextInputValue('additional') || null,
    };

    const ticketData = { type: 'appeal', userId: user.id, guildId, data, status: 'open', createdAt: Date.now(), claimedBy: null, involvedStaff: [], dmMessageIds: [], threadId: null, channelId: null, messageId: null };
    setTicket(refId, ticketData);

    const msg = await channel.send({ embeds: [buildAppealEmbed(refId, user.id, data)], components: buildActionRows(refId, 'appeal', { claimed: false, terminal: false }) });
    ticketData.channelId = channel.id; ticketData.messageId = msg.id;
    ticketData.threadId  = await createThread(msg, refId);
    setTicket(refId, ticketData);

    const dmSent = await sendDM(user.id, {
      embeds: [new EmbedBuilder().setTitle('✅  Appeal Submitted').setColor(0x57F287)
        .setDescription(`Your appeal has been submitted to the moderation team.\n\n**Reference ID:** \`${refId}\`\n\nKeep this ID — all updates will reference it.\n\nHave more to add? Reply to this DM at any time.`)
        .setTimestamp()]
    });

    return interaction.reply({
      content: dmSent
        ? `📬  Appeal submitted! Check your DMs for your reference ID (\`${refId}\`).`
        : `✅  Appeal submitted! Your reference ID is \`${refId}\`. *(Enable DMs from server members to receive future updates.)*`,
      flags: E,
    });
  }

  // ── Staff → Reporter message ──────────────────────────────────
  if (customId.startsWith('staffmsg_')) {
    const refId  = customId.slice('staffmsg_'.length);
    const ticket = getTicket(refId);
    if (!ticket) return interaction.reply({ content: '❌  Ticket not found.', flags: E });
    const content = interaction.fields.getTextInputValue('message');
    const sentMsg = await sendDMGetMsg(ticket.userId, {
      embeds: [new EmbedBuilder().setTitle(`📨  Message From Staff — \`${refId}\``).setColor(0x5865F2)
        .setDescription(content).setFooter({ text: 'Reply to this DM to respond — staff will be notified.' }).setTimestamp()]
    });
    if (!sentMsg) return interaction.reply({ content: '❌  Could not DM the reporter — their DMs may be closed.', flags: E });

    ticket.dmMessageIds = ticket.dmMessageIds || []; ticket.dmMessageIds.push(sentMsg.id);
    ticket.involvedStaff = ticket.involvedStaff || [];
    if (!ticket.involvedStaff.includes(interaction.user.id)) ticket.involvedStaff.push(interaction.user.id);
    setTicket(refId, ticket);

    if (ticket.threadId) {
      const thread = await client.channels.fetch(ticket.threadId).catch(() => null);
      if (thread) await thread.send({ embeds: [new EmbedBuilder().setColor(0x57F287).setDescription(content).setFooter({ text: `Staff → Reporter — ${interaction.user.tag}` }).setTimestamp()] }).catch(() => {});
    }
    return interaction.reply({ content: '✅  Message sent to the reporter.', flags: E });
  }

  // ── Resolution note modals ────────────────────────────────────
  if (customId.startsWith('handle_note_'))  return finalizeTicket(interaction, customId.slice('handle_note_'.length),  'handled');
  if (customId.startsWith('insuff_note_'))  return finalizeTicket(interaction, customId.slice('insuff_note_'.length),  'insufficient');
  if (customId.startsWith('accept_note_'))  return finalizeTicket(interaction, customId.slice('accept_note_'.length),  'accepted');
  if (customId.startsWith('deny_note_'))    return finalizeTicket(interaction, customId.slice('deny_note_'.length),    'denied');

  // ── Q&A: Question submission modal ───────────────────────────
  if (customId.startsWith('qa_question_')) {
    const anonymous = customId.slice('qa_question_'.length) === '1';
    const freshMember = await resolveFreshMember(interaction);

    // Re-check ban role at modal submit time (could change between button click & submit)
    const qa = getQaConfig(guildId);
    if (qa.banRoleId && freshMember?.roles.cache.has(qa.banRoleId))
      return interaction.reply({ content: '🚫  You are not allowed to submit questions.', flags: E });

    const qText = interaction.fields.getTextInputValue('question').trim().slice(0, 1000);
    if (!qText) return interaction.reply({ content: '❌  Question cannot be empty.', flags: E });

    const targetChannel = qa.targetChannelId ? await client.channels.fetch(qa.targetChannelId).catch(() => null) : null;
    if (!targetChannel) return interaction.reply({ content: '❌  Target channel not configured. Contact an administrator.', flags: E });

    const embed = new EmbedBuilder().setTitle('📩  New Question').setColor(0xFEE75C)
      .addFields(
        { name: 'Question',     value: qText,                                                 inline: false },
        { name: 'Anonymous',    value: anonymous ? '✅ Yes' : '❌ No',                         inline: true  },
        { name: 'Submitted By', value: `${interaction.user} (\`${interaction.user.tag}\`)`,   inline: true  },
      )
      .setThumbnail(interaction.user.displayAvatarURL({ dynamic: true }))
      .setFooter({ text: `User ID: ${interaction.user.id}` });

    const answerBtn = new ButtonBuilder().setCustomId(`qa_answer_${interaction.user.id}`).setLabel('Answer Question').setEmoji('✅').setStyle(ButtonStyle.Success);
    const denyBtn   = new ButtonBuilder().setCustomId(`qa_deny_${interaction.user.id}`).setLabel('Deny Question').setEmoji('❌').setStyle(ButtonStyle.Danger);
    await targetChannel.send({ embeds: [embed], components: [new ActionRowBuilder().addComponents(answerBtn, denyBtn)] });

    // Record cooldown only after successful post
    qaRecordSubmission(interaction.user.id);
    return interaction.reply({ content: '✅  Your question has been submitted!', flags: E });
  }

  // ── Q&A: Answer modal ─────────────────────────────────────────
  if (customId.startsWith('qa_answer_modal_')) {
    const submitterId = customId.slice('qa_answer_modal_'.length);
    const freshMember = await resolveFreshMember(interaction);
    if (!canModerateQa(freshMember, guildId))
      return interaction.reply({ content: '🚫  Your permission has been revoked.', flags: E });

    const answerText = interaction.fields.getTextInputValue('answer').trim().slice(0, 1000);
    if (!answerText) return interaction.reply({ content: '❌  Answer cannot be empty.', flags: E });

    // Get the question embed from the original message
    const srcEmbed = interaction.message?.embeds?.[0];
    const qText    = srcEmbed?.fields?.find(f => f.name === 'Question')?.value ?? '*(unknown)*';
    const isAnon   = srcEmbed?.fields?.find(f => f.name === 'Anonymous')?.value?.includes('Yes') ?? false;

    const qa = getQaConfig(guildId);
    const answerChannel = qa.answerChannelId ? await client.channels.fetch(qa.answerChannelId).catch(() => null) : null;
    if (!answerChannel) return interaction.reply({ content: '❌  Answer channel not configured.', flags: E });

    // Format quoted question like Python version
    const quotedQ = qText.split('\n').map(l => `> ${l}`).join('\n');
    const asker   = isAnon ? '*Anonymous User*' : `<@${submitterId}>`;
    await answerChannel.send(`## ✅ Question Answered! ✅\n${quotedQ}\n${answerText}\n${asker}`);

    // Update original message — mark answered, disable buttons
    if (interaction.message) {
      const updated = EmbedBuilder.from(srcEmbed).setTitle('📩  Question — Answered ✅').setColor(0x57F287)
        .addFields({ name: 'Answered By', value: `${interaction.user}`, inline: false });
      const disabledRow = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId(`qa_answer_${submitterId}`).setLabel('Answer Question').setEmoji('✅').setStyle(ButtonStyle.Success).setDisabled(true),
        new ButtonBuilder().setCustomId(`qa_deny_${submitterId}`).setLabel('Deny Question').setEmoji('❌').setStyle(ButtonStyle.Danger).setDisabled(true),
      );
      await interaction.message.edit({ embeds: [updated], components: [disabledRow] });
    }
    return interaction.reply({ content: '✅  Answer submitted!', flags: E });
  }

  // ── Q&A: Deny modal ───────────────────────────────────────────
  if (customId.startsWith('qa_deny_modal_')) {
    const submitterId = customId.slice('qa_deny_modal_'.length);
    const freshMember = await resolveFreshMember(interaction);
    if (!canModerateQa(freshMember, guildId))
      return interaction.reply({ content: '🚫  Your permission has been revoked.', flags: E });

    const reasonText = interaction.fields.getTextInputValue('reason')?.trim().slice(0, 500) || null;
    const srcEmbed   = interaction.message?.embeds?.[0];
    const qText      = srcEmbed?.fields?.find(f => f.name === 'Question')?.value ?? '*(unknown)*';

    // DM the submitter
    if (submitterId) {
      const dmEmbed = new EmbedBuilder().setTitle('❌  Your Question Was Denied').setColor(0xED4245)
        .addFields(
          { name: 'Your Question', value: qText,                            inline: false },
          { name: 'Reason',        value: reasonText || 'No reason provided.', inline: false },
        )
        .setFooter({ text: `Server: ${interaction.guild?.name ?? 'Unknown'}` });
      await tryDM(submitterId, { embeds: [dmEmbed] });
    }

    // Update original message — mark denied, disable buttons
    if (interaction.message && srcEmbed) {
      const updatedFields = srcEmbed.fields.filter(f => !['Denied By','Reason'].includes(f.name));
      const updated = new EmbedBuilder().setTitle('📩  Question — Denied ❌').setColor(0xED4245);
      for (const f of updatedFields) updated.addFields({ name: f.name, value: f.value, inline: f.inline });
      updated.addFields({ name: 'Denied By', value: `${interaction.user}`, inline: false });
      if (reasonText) updated.addFields({ name: 'Reason', value: reasonText, inline: false });
      if (srcEmbed.thumbnail) updated.setThumbnail(srcEmbed.thumbnail.url);
      if (srcEmbed.footer)    updated.setFooter({ text: srcEmbed.footer.text });
      const disabledRow = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId(`qa_answer_${submitterId}`).setLabel('Answer Question').setEmoji('✅').setStyle(ButtonStyle.Success).setDisabled(true),
        new ButtonBuilder().setCustomId(`qa_deny_${submitterId}`).setLabel('Deny Question').setEmoji('❌').setStyle(ButtonStyle.Danger).setDisabled(true),
      );
      await interaction.message.edit({ embeds: [updated], components: [disabledRow] });
    }
    return interaction.reply({ content: '✅  Question denied, user notified.', flags: E });
  }
}

// ─────────────────────────────────────────────────────────────
//  Start
// ─────────────────────────────────────────────────────────────
client.login(process.env.TOKEN);
