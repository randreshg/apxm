use unicode_width::UnicodeWidthChar;

/// Multi-line text buffer used by the chat composer.
#[derive(Debug, Default, Clone)]
pub struct InputBuffer {
    content: String,
    cursor: usize,
}

impl InputBuffer {
    pub fn clear(&mut self) {
        self.content.clear();
        self.cursor = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }

    pub fn as_str(&self) -> &str {
        &self.content
    }

    pub fn take(&mut self) -> String {
        let mut text = String::new();
        std::mem::swap(&mut self.content, &mut text);
        self.cursor = 0;
        text
    }

    pub fn insert_char(&mut self, ch: char) {
        self.content.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub fn insert_str(&mut self, text: &str) {
        for ch in text.chars() {
            self.insert_char(ch);
        }
    }

    pub fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let len = self.prev_char_len();
        if len > 0 {
            self.cursor -= len;
            self.content.drain(self.cursor..self.cursor + len);
        }
    }

    pub fn delete(&mut self) {
        if self.cursor >= self.content.len() {
            return;
        }
        let len = self.next_char_len();
        if len > 0 {
            self.content.drain(self.cursor..self.cursor + len);
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let len = self.prev_char_len();
        self.cursor = self.cursor.saturating_sub(len);
    }

    pub fn move_right(&mut self) {
        let len = self.next_char_len();
        self.cursor = (self.cursor + len).min(self.content.len());
    }

    pub fn move_to_start_of_line(&mut self) {
        if self.cursor == 0 {
            return;
        }
        if let Some(idx) = self.content[..self.cursor].rfind('\n') {
            self.cursor = idx + 1;
        } else {
            self.cursor = 0;
        }
    }

    pub fn move_to_end_of_line(&mut self) {
        if let Some(idx) = self.content[self.cursor..].find('\n') {
            self.cursor += idx;
        } else {
            self.cursor = self.content.len();
        }
    }

    pub fn move_up(&mut self) {
        let (row, col) = self.cursor_position_chars();
        if row == 0 {
            self.cursor = 0;
            return;
        }

        let lines: Vec<&str> = self.content.split('\n').collect();
        let target_line = row.saturating_sub(1);
        let mut new_cursor = 0usize;
        for idx in 0..target_line {
            new_cursor += lines.get(idx).map(|line| line.len() + 1).unwrap_or(0);
        }
        let target = lines
            .get(target_line)
            .map(|line| Self::cursor_offset_for_column(line, col))
            .unwrap_or(0);
        self.cursor = (new_cursor + target).min(self.content.len());
    }

    pub fn move_down(&mut self) {
        let (row, col) = self.cursor_position_chars();
        let lines: Vec<&str> = self.content.split('\n').collect();
        if row + 1 >= lines.len() {
            self.cursor = self.content.len();
            return;
        }

        let mut new_cursor = 0usize;
        for idx in 0..=row {
            new_cursor += lines.get(idx).map(|line| line.len() + 1).unwrap_or(0);
        }
        let target = lines
            .get(row + 1)
            .map(|line| Self::cursor_offset_for_column(line, col))
            .unwrap_or(0);
        self.cursor = (new_cursor + target).min(self.content.len());
    }

    pub fn line_count(&self) -> usize {
        if self.content.is_empty() {
            1
        } else {
            self.content.split('\n').count()
        }
    }

    pub fn cursor_position(&self) -> (u16, u16) {
        let (row, col) = self.cursor_position_chars();
        (row as u16, col as u16)
    }

    fn cursor_position_chars(&self) -> (usize, usize) {
        let head = &self.content[..self.cursor];
        let row = head.bytes().filter(|b| *b == b'\n').count();
        let col = head
            .rsplit_once('\n')
            .map(|(_, rest)| rest.chars().count())
            .unwrap_or_else(|| head.chars().count());
        (row, col)
    }

    fn prev_char_len(&self) -> usize {
        self.content[..self.cursor]
            .chars()
            .next_back()
            .map(|ch| ch.len_utf8())
            .unwrap_or(0)
    }

    fn next_char_len(&self) -> usize {
        self.content[self.cursor..]
            .chars()
            .next()
            .map(|ch| ch.len_utf8())
            .unwrap_or(0)
    }

    fn cursor_offset_for_column(line: &str, target_col: usize) -> usize {
        let mut col = 0usize;
        let mut byte_idx = 0usize;
        for ch in line.chars() {
            if col >= target_col {
                break;
            }
            col += UnicodeWidthChar::width(ch).unwrap_or(1);
            byte_idx += ch.len_utf8();
        }
        byte_idx
    }
}
