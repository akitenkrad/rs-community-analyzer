//! End-to-end synthetic demo: build a tiny org communication snapshot,
//! compute H1–H5 with `WhitespaceMorphology`, print the summary.
//!
//! Run with: `cargo run --example basic`.

use chrono::{DateTime, Duration, TimeZone, Utc};

use community_analyzer::{
    build_summary, compute_h1, compute_h2, compute_h3, compute_h4, compute_h5, AnalysisInput,
    Channel, ChannelCategory, CommConfig, Message, Reaction, Role, User, WhitespaceMorphology,
};

fn ts(secs: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(secs, 0).single().unwrap()
}

fn msg(
    id: &str,
    ch: &str,
    author: &str,
    text: &str,
    t: DateTime<Utc>,
    root: Option<&str>,
) -> Message {
    Message {
        id: id.into(),
        channel_id: ch.into(),
        author_id: author.into(),
        text: text.into(),
        timestamp: t,
        thread_root_id: root.map(String::from),
        reaction_count: 0,
    }
}

fn main() {
    let cfg = CommConfig::default();

    let channels = vec![
        Channel {
            id: "C_DEC".into(),
            name: "decisions".into(),
            category: Some(ChannelCategory::Official),
            is_decision_channel: true,
        },
        Channel {
            id: "C_TECH".into(),
            name: "engineering".into(),
            category: Some(ChannelCategory::Tech),
            is_decision_channel: false,
        },
        Channel {
            id: "C_CHAT".into(),
            name: "random".into(),
            category: Some(ChannelCategory::Casual),
            is_decision_channel: false,
        },
    ];

    let users = vec![
        User {
            id: "U_LEAD".into(),
            display_name: "Lead".into(),
            role: Role::Manager,
        },
        User {
            id: "U_ARCH".into(),
            display_name: "Architect".into(),
            role: Role::Lead,
        },
        User {
            id: "U_DEV1".into(),
            display_name: "Dev1".into(),
            role: Role::Staff,
        },
        User {
            id: "U_DEV2".into(),
            display_name: "Dev2".into(),
            role: Role::Staff,
        },
        User {
            id: "U_QA".into(),
            display_name: "QA".into(),
            role: Role::Staff,
        },
    ];

    // 30 messages forming 5 threads across the 3 channels.
    let base = ts(1_735_000_000);
    let mut messages: Vec<Message> = Vec::new();
    let add = |v: &mut Vec<Message>, id, ch, u, text, off, root| {
        v.push(msg(id, ch, u, text, base + Duration::seconds(off), root));
    };

    // Thread 1 in C_DEC: lead proposes; team agrees (surface-agreement).
    add(
        &mut messages,
        "T1_M0",
        "C_DEC",
        "U_LEAD",
        "提案: 新アーキ採用",
        0,
        None,
    );
    add(
        &mut messages,
        "T1_M1",
        "C_DEC",
        "U_ARCH",
        "承知しました",
        60,
        Some("T1_M0"),
    );
    add(
        &mut messages,
        "T1_M2",
        "C_DEC",
        "U_DEV1",
        "賛成です",
        120,
        Some("T1_M0"),
    );
    add(
        &mut messages,
        "T1_M3",
        "C_DEC",
        "U_DEV2",
        "了解です",
        180,
        Some("T1_M0"),
    );
    add(
        &mut messages,
        "T1_M4",
        "C_DEC",
        "U_LEAD",
        "結論: 採用で決定",
        240,
        Some("T1_M0"),
    );

    // Thread 2 in C_TECH: incident; commitment then completion.
    add(
        &mut messages,
        "T2_M0",
        "C_TECH",
        "U_DEV1",
        "障害が発生しています",
        600,
        None,
    );
    add(
        &mut messages,
        "T2_M1",
        "C_TECH",
        "U_DEV2",
        "対応します",
        700,
        Some("T2_M0"),
    );
    add(
        &mut messages,
        "T2_M2",
        "C_TECH",
        "U_DEV2",
        "修正しました",
        4300,
        Some("T2_M0"),
    );
    add(
        &mut messages,
        "T2_M3",
        "C_TECH",
        "U_QA",
        "確認しました",
        4400,
        Some("T2_M0"),
    );

    // Thread 3 in C_DEC: dissent exists but is dampened.
    add(
        &mut messages,
        "T3_M0",
        "C_DEC",
        "U_LEAD",
        "次の方針案",
        7200,
        None,
    );
    add(
        &mut messages,
        "T3_M1",
        "C_DEC",
        "U_DEV1",
        "とりあえず承知しました",
        7260,
        Some("T3_M0"),
    );
    add(
        &mut messages,
        "T3_M2",
        "C_DEC",
        "U_ARCH",
        "ただ懸念があります",
        7320,
        Some("T3_M0"),
    );
    add(
        &mut messages,
        "T3_M3",
        "C_DEC",
        "U_LEAD",
        "結論: そのまま進める",
        7440,
        Some("T3_M0"),
    );

    // Thread 4 in C_TECH: open exploration thread (no conclusion marker).
    add(
        &mut messages,
        "T4_M0",
        "C_TECH",
        "U_ARCH",
        "提案: 別案として B はどうでしょう ❓",
        10000,
        None,
    );
    add(
        &mut messages,
        "T4_M1",
        "C_TECH",
        "U_DEV1",
        "アイデアとして案 C もあります",
        10100,
        Some("T4_M0"),
    );
    add(
        &mut messages,
        "T4_M2",
        "C_TECH",
        "U_DEV2",
        "別の見方ですが…",
        10200,
        Some("T4_M0"),
    );

    // Thread 5 in C_CHAT: casual back-and-forth (post-meeting dissent setup).
    add(
        &mut messages,
        "T5_M0",
        "C_CHAT",
        "U_DEV1",
        "ただ、本当は反対だった",
        11000,
        None,
    );
    add(
        &mut messages,
        "T5_M1",
        "C_CHAT",
        "U_DEV2",
        "わかります",
        11100,
        Some("T5_M0"),
    );

    // Some lone messages spread across channels.
    add(
        &mut messages,
        "X1",
        "C_TECH",
        "U_DEV2",
        "雑にメモ FYI",
        12000,
        None,
    );
    add(
        &mut messages,
        "X2",
        "C_TECH",
        "U_QA",
        "確認します",
        12500,
        None,
    );
    add(
        &mut messages,
        "X3",
        "C_CHAT",
        "U_DEV1",
        "雑談です",
        13000,
        None,
    );
    add(
        &mut messages,
        "X4",
        "C_DEC",
        "U_LEAD",
        "通常発言",
        14000,
        None,
    );
    add(
        &mut messages,
        "X5",
        "C_DEC",
        "U_ARCH",
        "確認しました",
        15000,
        Some("X4"),
    );
    add(
        &mut messages,
        "X6",
        "C_TECH",
        "U_ARCH",
        "完了しました",
        16000,
        None,
    );
    add(
        &mut messages,
        "X7",
        "C_TECH",
        "U_DEV1",
        "違ったらすみません",
        17000,
        None,
    );
    add(
        &mut messages,
        "X8",
        "C_CHAT",
        "U_DEV2",
        "とりあえず",
        18000,
        None,
    );
    add(
        &mut messages,
        "X9",
        "C_TECH",
        "U_QA",
        "通常メッセ",
        19000,
        None,
    );
    add(
        &mut messages,
        "X10",
        "C_TECH",
        "U_LEAD",
        "対応します",
        20000,
        None,
    );
    add(
        &mut messages,
        "X11",
        "C_TECH",
        "U_LEAD",
        "対応しました",
        23600,
        None,
    ); // commitment -> action 1h
    add(
        &mut messages,
        "X12",
        "C_DEC",
        "U_ARCH",
        "賛成です",
        24000,
        None,
    );

    // Reactions: agreement on a few decision-channel messages.
    let reactions = vec![
        Reaction {
            message_id: "T1_M0".into(),
            channel_id: "C_DEC".into(),
            user_id: "U_DEV1".into(),
            emoji_name: "thumbsup".into(),
        },
        Reaction {
            message_id: "T1_M0".into(),
            channel_id: "C_DEC".into(),
            user_id: "U_DEV2".into(),
            emoji_name: "+1".into(),
        },
        Reaction {
            message_id: "X4".into(),
            channel_id: "C_DEC".into(),
            user_id: "U_QA".into(),
            emoji_name: "ok".into(),
        },
        Reaction {
            message_id: "T1_M2".into(),
            channel_id: "C_DEC".into(),
            user_id: "U_LEAD".into(),
            emoji_name: "clap".into(),
        },
    ];

    let input = AnalysisInput {
        messages: &messages,
        channels: &channels,
        users: &users,
        reactions: &reactions,
        config: &cfg,
    };

    let morph = WhitespaceMorphology;

    let h1 = compute_h1(&input).expect("compute_h1");
    let h2 = compute_h2(&input).expect("compute_h2");
    let h3 = compute_h3(&input).expect("compute_h3");
    let h4 = compute_h4(&input).expect("compute_h4");
    let h5 = compute_h5(&input, &morph).expect("compute_h5");
    let summary = build_summary(Some(&h1), Some(&h2), Some(&h3), Some(&h4), Some(&h5));

    println!("=== community-analyzer basic example ===");
    println!("messages: {}", messages.len());
    println!("channels: {}", channels.len());
    println!("users:    {}", users.len());
    println!();
    println!("[H1] thread_count          = {}", h1.thread_count);
    println!(
        "[H1] silence ratio (mgr/staff) = {:.2}",
        h1.silence_after_manager.ratio
    );
    println!("[H2] hedging_rate          = {:.4}", h2.hedging_rate);
    println!("[H2] self_defense_rate     = {:.4}", h2.self_defense_rate);
    println!(
        "[H2] escalation_delay(min) = {:.2}",
        h2.escalation_delay_minutes
    );
    println!(
        "[H3] reply_concentration_gini = {:.4}",
        h3.reply_concentration_gini
    );
    println!("[H3] hierarchy_score       = {:.4}", h3.hierarchy_score);
    println!(
        "[H3] manager_reaction_rate = {:.4}",
        h3.manager_reaction_rate
    );
    println!(
        "[H4] surface_agreement_rate = {:.4}",
        h4.surface_agreement_rate
    );
    println!(
        "[H4] post_meeting_dissent_rate = {:.4}",
        h4.post_meeting_dissent_rate
    );
    println!(
        "[H4] execution_delay(hr)   = {:.2}",
        h4.execution_delay_hours
    );
    println!(
        "[H5] unresolved_thread_ratio = {:.4}",
        h5.unresolved_thread_ratio
    );
    println!(
        "[H5] novel_vocab months    = {}",
        h5.novel_vocabulary_rate_monthly.len()
    );
    println!();
    println!("=== Summary ===");
    println!("overall_health_score = {:.2}", summary.overall_health_score);
    println!("red    flags: {}", summary.red_flags.len());
    println!("yellow flags: {}", summary.yellow_flags.len());
    println!("green  signals: {}", summary.green_signals.len());
}
