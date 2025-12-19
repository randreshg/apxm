use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use unicode_width::UnicodeWidthStr;

use crate::{commands, storage::Message};

use super::app::AppState;

pub fn render(f: &mut Frame<'_>, app: &AppState) {
    let size = f.size();
    render_center(f, app, size);
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
        // Each suggestion renders as two lines (usage + description) roughly; clamp height.
        let suggestion_height = (slash_matches.len() as u16 * 2).clamp(2, 12);
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
        // Pass the currently selected suggestion index (if any) so the menu can highlight it.
        render_slash_suggestions(
            f,
            &slash_matches,
            app.input.slash_selected_index(),
            chunks[idx],
        );
        idx += 1;
    }
    render_status(f, app, chunks[idx]);
}

fn render_header(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let id = short_id(&app.current_session_id);

    let mut header_spans = vec![
        Span::styled(
            "APXM",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled("·", Style::default().fg(Color::DarkGray)),
        Span::raw(" Session "),
        Span::styled(id, Style::default().fg(Color::Green)),
    ];

    if !app.messages.is_empty() {
        header_spans.push(Span::raw("  "));
        header_spans.push(Span::styled("·", Style::default().fg(Color::DarkGray)));
        header_spans.push(Span::raw(" "));
        header_spans.push(Span::styled(
            format!("{} msg", app.messages.len()),
            Style::default().fg(Color::Gray),
        ));
    }

    if app.is_processing {
        header_spans.push(Span::raw("  "));
        header_spans.push(Span::styled("·", Style::default().fg(Color::DarkGray)));
        header_spans.push(Span::raw(" "));
        header_spans.push(Span::styled(
            "Processing",
            Style::default().fg(Color::Yellow),
        ));
    }

    let header_line = Line::from(header_spans);
    let paragraph = Paragraph::new(Text::from(header_line)).alignment(Alignment::Left);
    f.render_widget(paragraph, area);
}

fn render_chat(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let mut lines = Vec::new();

    // Render existing messages
    for message in &app.messages {
        lines.extend(format_message(message));
    }

    // Render streaming response with enhanced visuals
    if let Some(stream) = &app.streaming {
        lines.push(Line::from(vec![
            Span::styled(
                "APXM",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("{} streaming", spinner(app.tick_count() as usize)),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]));

        if !stream.buffer.is_empty() {
            lines.extend(format_markdown(&stream.buffer));
            // Add cursor indicator for active streaming
            lines.push(Line::from(Span::styled(
                "▊",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::SLOW_BLINK),
            )));
        }
    }

    // Empty state with helpful message
    if lines.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Welcome to APXM",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(
            "Ask me to help you build autonomous agents in natural language.",
        ));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "Examples:",
            Style::default().fg(Color::Yellow),
        )]));
        lines.push(Line::from("  • Create an agent that greets users"));
        lines.push(Line::from("  • Build a search agent with web access"));
        lines.push(Line::from("  • Make an agent that reflects on its actions"));
    }

    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default())
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

    let (row, col) = app.input.cursor_position();
    let cursor_x = area.x + 1 + col;
    let cursor_y = area.y + 1 + row;
    f.set_cursor(cursor_x, cursor_y);
}

fn render_status(f: &mut Frame<'_>, app: &AppState, area: Rect) {
    let mut segments = Vec::new();

    if app.streaming.is_some() {
        segments.push(Span::styled(
            spinner(app.tick_count() as usize),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        segments.push(Span::raw(" "));

        if let Some(stream) = &app.streaming {
            if let Some(status) = &stream.last_status {
                segments.push(Span::styled(
                    status.clone(),
                    Style::default().fg(Color::Yellow),
                ));
            } else {
                segments.push(Span::styled(
                    "Processing...",
                    Style::default().fg(Color::Yellow),
                ));
            }
        }
    } else if app.is_processing {
        segments.push(Span::styled("⏳ ", Style::default().fg(Color::Yellow)));
        segments.push(Span::styled(
            "Processing request...",
            Style::default().fg(Color::Yellow),
        ));
    } else if let Some(msg) = &app.status_line {
        let (icon, color) = if msg.contains("error")
            || msg.contains("Error")
            || msg.contains("failed")
        {
            ("✗ ", Color::Red)
        } else if msg.contains("success") || msg.contains("Success") || msg.contains("completed") {
            ("✓ ", Color::Green)
        } else {
            ("ℹ ", Color::Cyan)
        };

        segments.push(Span::styled(icon, Style::default().fg(color)));
        segments.push(Span::styled(msg.clone(), Style::default().fg(Color::Gray)));
    } else {
        segments.push(Span::styled("Ready", Style::default().fg(Color::Green)));
        segments.push(Span::raw("  ·  "));
        segments.push(Span::styled("Ctrl+1-5", Style::default().fg(Color::Yellow)));
        segments.push(Span::raw(" views   "));
        segments.push(Span::styled("Ctrl+P", Style::default().fg(Color::Yellow)));
        segments.push(Span::raw(" panels   "));
        segments.push(Span::styled("Tab", Style::default().fg(Color::Yellow)));
        segments.push(Span::raw(" focus"));
    }

    let paragraph = Paragraph::new(Line::from(segments));
    f.render_widget(paragraph, area);
}

fn render_slash_suggestions(
    f: &mut Frame<'_>,
    entries: &[&commands::SlashMetadata],
    selected: Option<usize>,
    area: Rect,
) {
    if entries.is_empty() {
        return;
    }

    // Compute a reasonable column width for the usage column using display width.
    let mut usage_width: usize = entries.iter().map(|e| e.usage.width()).max().unwrap_or(0);
    // Clamp so the usage column doesn't consume the whole area
    usage_width = usage_width.min(30);

    let mut lines = Vec::new();
    for (idx, entry) in entries.iter().enumerate() {
        if idx > 0 {
            lines.push(Line::from(""));
        }

        let is_selected = selected.map(|s| s == idx).unwrap_or(false);

        // Prepare usage span (fixed-width padded to usage_width)
        let usage_text = if entry.usage.width() >= usage_width {
            entry.usage.to_string()
        } else {
            format!("{:width$}", entry.usage, width = usage_width)
        };

        let usage_span = if is_selected {
            Span::styled(
                usage_text,
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(
                usage_text,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
        };

        // Description shown to the right of usage; highlight if selected.
        let desc_span = if is_selected {
            Span::styled(
                format!(" {}", entry.description),
                Style::default().fg(Color::Black).bg(Color::Yellow),
            )
        } else {
            Span::raw(format!(" {}", entry.description))
        };

        lines.push(Line::from(vec![usage_span, desc_span]));
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
    let mut code_lang = String::new();

    for raw in content.lines() {
        let trimmed = raw.trim_start();

        // Handle code blocks with language detection
        if trimmed.starts_with("```") {
            in_code = !in_code;
            if in_code {
                code_lang = trimmed.trim_start_matches("```").to_string();
                let code_label = if code_lang.is_empty() {
                    "code".to_string()
                } else {
                    code_lang.clone()
                };
                lines.push(Line::from(vec![
                    Span::styled("╭─ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        code_label,
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
            } else {
                lines.push(Line::from(Span::styled(
                    "╰─",
                    Style::default().fg(Color::DarkGray),
                )));
                code_lang.clear();
            }
            continue;
        }

        if in_code {
            lines.push(Line::from(vec![
                Span::styled("│ ", Style::default().fg(Color::DarkGray)),
                Span::styled(raw.to_string(), Style::default().fg(Color::Cyan)),
            ]));
            continue;
        }

        // Bullet lists with better styling
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
            lines.push(Line::from(vec![
                Span::styled("  • ", Style::default().fg(Color::Yellow)),
                Span::raw(trimmed[2..].to_string()),
            ]));
            continue;
        }

        // Numbered lists
        if let Some(rest) = trimmed.strip_prefix(|c: char| c.is_ascii_digit())
            && rest.starts_with(". ")
        {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} ", trimmed.split(". ").next().unwrap_or("")),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(rest.trim_start_matches(". ").to_string()),
            ]));
            continue;
        }

        // Headers with different levels
        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|&c| c == '#').count();
            let content = trimmed.trim_start_matches('#').trim();
            let (color, modifier) = match level {
                1 => (Color::Yellow, Modifier::BOLD),
                2 => (Color::Cyan, Modifier::BOLD),
                _ => (Color::Blue, Modifier::BOLD),
            };

            lines.push(Line::from(Span::styled(
                content.to_string(),
                Style::default().fg(color).add_modifier(modifier),
            )));
            continue;
        }

        // Blockquotes
        if let Some(stripped) = trimmed.strip_prefix("> ") {
            lines.push(Line::from(vec![
                Span::styled("▎ ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    stripped.to_string(),
                    Style::default()
                        .fg(Color::Gray)
                        .add_modifier(Modifier::ITALIC),
                ),
            ]));
            continue;
        }

        // Empty lines and regular text
        if trimmed.is_empty() {
            lines.push(Line::from(""));
        } else {
            // Simple inline code detection (backticks)
            if trimmed.contains('`') {
                let mut spans = Vec::new();
                let mut in_inline_code = false;
                let mut current = String::new();

                for ch in trimmed.chars() {
                    if ch == '`' {
                        if in_inline_code {
                            spans.push(Span::styled(
                                current.clone(),
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD),
                            ));
                            current.clear();
                        } else if !current.is_empty() {
                            spans.push(Span::raw(current.clone()));
                            current.clear();
                        }
                        in_inline_code = !in_inline_code;
                    } else {
                        current.push(ch);
                    }
                }

                if !current.is_empty() {
                    if in_inline_code {
                        spans.push(Span::styled(
                            current,
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ));
                    } else {
                        spans.push(Span::raw(current));
                    }
                }

                lines.push(Line::from(spans));
            } else {
                lines.push(Line::from(trimmed.to_string()));
            }
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
    const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    SPINNER_FRAMES[step % SPINNER_FRAMES.len()]
}
