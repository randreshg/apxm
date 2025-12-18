use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use crate::{commands, storage::Message};

use super::app::{AppState, FocusTarget, ViewMode};

pub fn render(f: &mut Frame<'_>, app: &AppState) {
    let size = f.size();
    let right_width = if app.right_sidebar_visible { 34 } else { 0 };

    let chunks = if app.right_sidebar_visible && right_width > 0 {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(50), Constraint::Length(right_width)])
            .split(size)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(50)])
            .split(size)
    };

    render_center(f, app, chunks[0]);

    if app.right_sidebar_visible && chunks.len() > 1 {
        render_specialist(f, app, chunks[1]);
    }
}

fn render_center(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let input_height = (app.input.line_count() as u16).clamp(3, 8) + 2;
    let slash_matches = commands::slash_suggestions(app.input.as_str());
    let show_suggestions = !slash_matches.is_empty();

    let mut constraints = vec![
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(input_height),
    ];
    if show_suggestions {
        let suggestion_height = (slash_matches.len() as u16 * 2).max(2).min(10);
        constraints.push(Constraint::Length(suggestion_height));
    }
    constraints.push(Constraint::Length(1));

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let mut idx = 0;
    render_header(f, app, chunks[idx]);
    idx += 1;
    render_chat(f, app, chunks[idx]);
    idx += 1;
    render_input(f, app, chunks[idx]);
    idx += 1;
    if show_suggestions {
        render_slash_suggestions(f, &slash_matches, chunks[idx]);
        idx += 1;
    }
    render_status(f, app, chunks[idx]);
}

fn render_header(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let id = short_id(&app.current_session_id);
    let header_line = Line::from(vec![
        Span::styled(
            format!("Session {id}"),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  ·  "),
        Span::styled(
            format!("{} messages", app.messages.len()),
            Style::default().fg(Color::Gray),
        ),
    ]);

    let paragraph = Paragraph::new(Text::from(header_line)).alignment(Alignment::Left);
    f.render_widget(paragraph, area);
}

fn render_chat(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let mut lines = Vec::new();
    for message in &app.messages {
        lines.extend(format_message(message));
    }
    if let Some(stream) = &app.streaming {
        lines.push(Line::from(vec![Span::styled(
            "APXM (streaming)…",
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::ITALIC),
        )]));
        if !stream.buffer.is_empty() {
            lines.extend(format_markdown(&stream.buffer));
        }
    }
    if lines.is_empty() {
        lines.push(Line::from("No messages yet. Ask APxM for help."));
    }

    let style = if app.focus == FocusTarget::Chat {
        Style::default()
    } else {
        Style::default().fg(Color::Gray)
    };

    let paragraph = Paragraph::new(Text::from(lines))
        .style(style)
        .wrap(Wrap { trim: false })
        .scroll((app.chat_scroll, 0));

    f.render_widget(paragraph, area);
}

fn render_input(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(app.input.as_str())
        .block(block)
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, area);

    if matches!(app.focus, FocusTarget::Chat) {
        let (row, col) = app.input.cursor_position();
        let cursor_x = area.x + 1 + col;
        let cursor_y = area.y + 1 + row;
        f.set_cursor(cursor_x, cursor_y);
    }
}

fn render_status(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let mut segments = Vec::new();
    if let Some(msg) = &app.status_line {
        segments.push(Span::styled(msg.clone(), Style::default().fg(Color::Gray)));
    } else {
        segments.push(Span::raw(
            "Ctrl+N new session   ·   Ctrl+P toggle panels   ·   Tab move focus",
        ));
    }
    if app.streaming.is_some() {
        segments.push(Span::raw("   "));
        segments.push(Span::styled(
            spinner(app.tick_count() as usize),
            Style::default().fg(Color::Yellow),
        ));
        segments.push(Span::raw(" generating"));
    }

    let paragraph = Paragraph::new(Line::from(segments));
    f.render_widget(paragraph, area);
}

fn render_slash_suggestions(f: &mut Frame<'_>, entries: &[&commands::SlashMetadata], area: Rect) {
    if entries.is_empty() {
        return;
    }

    let mut lines = Vec::new();
    for (idx, entry) in entries.iter().enumerate() {
        if idx > 0 {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            entry.usage,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::raw(format!("  {}", entry.description))));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title("Slash commands");
    let paragraph = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: true });
    f.render_widget(paragraph, area);
}

const VIEW_TABS: &[(ViewMode, &str)] = &[
    (ViewMode::Minimal, "Shortcuts"),
    (ViewMode::Compiler, "Compiler"),
    (ViewMode::Dag, "Plan"),
    (ViewMode::Aam, "AAM"),
    (ViewMode::Full, "Plan+Build"),
];

fn render_specialist(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(3)])
        .split(area);

    let mut tab_spans = Vec::new();
    for (idx, (mode, label)) in VIEW_TABS.iter().enumerate() {
        if idx > 0 {
            tab_spans.push(Span::raw("  "));
        }
        let style = if app.view_mode == *mode {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        tab_spans.push(Span::styled(*label, style));
    }
    let tabs = Paragraph::new(Line::from(tab_spans));
    f.render_widget(tabs, chunks[0]);

    match app.view_mode {
        ViewMode::Minimal => render_shortcuts(f, app, chunks[1]),
        ViewMode::Compiler => render_compiler(f, app, chunks[1]),
        ViewMode::Dag => render_plan(f, app, chunks[1]),
        ViewMode::Aam => render_aam(f, app, chunks[1]),
        ViewMode::Full => render_full_stack(f, app, chunks[1]),
    }
}

fn render_shortcuts(f: &mut Frame<'_>, _app: &AppState, area: Rect) {
    let shortcuts = vec![
        "Ctrl+N  new session",
        "Ctrl+R  refresh sessions",
        "Ctrl+B  toggle left",
        "Ctrl+P  toggle right",
        "Ctrl+1-5 switch views",
        "Tab     cycle focus",
    ];
    let text = shortcuts
        .into_iter()
        .map(|entry| Line::from(entry.to_string()))
        .collect::<Vec<_>>();

    let paragraph = Paragraph::new(Text::from(text))
        .style(Style::default().fg(Color::Gray).add_modifier(Modifier::DIM));
    f.render_widget(paragraph, area);
}

fn render_compiler(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let mut lines = Vec::new();
    if let Some(code) = &app.specialized.dsl_code {
        lines.push(Line::from(vec![Span::styled(
            "Recent DSL:",
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        )]));
        for snippet in code.lines().take(40) {
            lines.push(Line::from(format!("  {snippet}")));
        }
    }
    if !app.specialized.compiler_log.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Diagnostics:",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )));
        for msg in &app.specialized.compiler_log {
            lines.push(Line::from(format!("  {msg}")));
        }
    }
    if !app.specialized.execution_results.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Execution Tokens:",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )));
        for row in &app.specialized.execution_results {
            lines.push(Line::from(format!("  {row}")));
        }
    }

    if lines.is_empty() {
        lines.push(Line::from("Execute a plan to view compiler output."));
    }

    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn render_plan(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let mut lines = Vec::new();
    if let Some(plan) = &app.specialized.plan {
        lines.push(Line::from(Span::styled(
            &plan.result,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )));
        for (idx, step) in plan.plan.iter().enumerate() {
            let mut content = format!("{}. {}", idx + 1, step.description);
            if !step.dependencies.is_empty() {
                content.push_str(&format!("  (deps: {})", step.dependencies.join(", ")));
            }
            lines.push(Line::from(content));
        }
    } else {
        lines.push(Line::from("No plan yet. Your next response will map here."));
    }

    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn render_aam(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let aam = &app.specialized.aam;
    let sections = [
        ("Goals", &aam.goals),
        ("Beliefs", &aam.beliefs),
        ("Memory", &aam.episodic_memory),
        ("Capabilities", &aam.capabilities),
    ];
    let mut lines = Vec::new();
    for (title, entries) in sections {
        lines.push(Line::from(Span::styled(
            title,
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        )));
        if entries.is_empty() {
            lines.push(Line::from("  (empty)"));
        } else {
            for entry in entries {
                lines.push(Line::from(format!("  {entry}")));
            }
        }
        lines.push(Line::from(""));
    }

    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn render_full_stack(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);
    render_plan(f, app, chunks[0]);
    render_compiler(f, app, chunks[1]);
}

fn format_message(message: &Message) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let header = Line::from(vec![
        Span::styled(role_label(&message.role), role_style(&message.role)),
        Span::raw("  "),
        Span::styled(
            message.timestamp.format("%H:%M").to_string(),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    lines.push(header);
    lines.extend(format_markdown(&message.content));
    lines.push(Line::from(""));
    lines
}

fn format_markdown(content: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code = false;
    for raw in content.lines() {
        if raw.trim_start().starts_with("```") {
            in_code = !in_code;
            lines.push(Line::from(Span::styled(
                raw.to_string(),
                Style::default().fg(Color::Yellow),
            )));
            continue;
        }

        if in_code {
            lines.push(Line::from(Span::styled(
                raw.to_string(),
                Style::default().fg(Color::Cyan),
            )));
            continue;
        }

        let trimmed = raw.trim_start();
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::Gray)),
                Span::raw(trimmed[2..].to_string()),
            ]));
            continue;
        }

        if trimmed.starts_with('#') {
            let content = trimmed.trim_start_matches('#').trim();
            lines.push(Line::from(Span::styled(
                content.to_string(),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }

        if trimmed.is_empty() {
            lines.push(Line::from(""));
        } else {
            lines.push(Line::from(trimmed.to_string()));
        }
    }
    if content.is_empty() {
        lines.push(Line::from(""));
    }
    lines
}

fn role_label(role: &str) -> String {
    match role {
        "assistant" => "APXM".to_string(),
        "system" => "System".to_string(),
        "user" => "You".to_string(),
        other => other.to_string(),
    }
}

fn role_style(role: &str) -> Style {
    match role {
        "assistant" => Style::default().fg(Color::Blue),
        "system" => Style::default().fg(Color::Yellow),
        "user" => Style::default().fg(Color::Green),
        _ => Style::default().fg(Color::White),
    }
}

fn short_id(id: &str) -> String {
    let len = id.len().min(8);
    id[..len].to_string()
}

fn spinner(step: usize) -> &'static str {
    match step % 4 {
        0 => "-",
        1 => "\\",
        2 => "|",
        _ => "/",
    }
}
