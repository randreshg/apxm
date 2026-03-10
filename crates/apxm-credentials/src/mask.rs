/// Mask an API key for display, showing first 4 and last 4 characters.
pub fn mask_key(key: &str) -> String {
    if key.len() <= 8 {
        "****".to_string()
    } else {
        format!("{}...{}", &key[..4], &key[key.len() - 4..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_long_key() {
        assert_eq!(mask_key("sk-proj-abc123xyz"), "sk-p...3xyz");
    }

    #[test]
    fn test_mask_short_key() {
        assert_eq!(mask_key("abc"), "****");
    }

    #[test]
    fn test_mask_exactly_8() {
        assert_eq!(mask_key("12345678"), "****");
    }

    #[test]
    fn test_mask_9_chars() {
        assert_eq!(mask_key("123456789"), "1234...6789");
    }
}
