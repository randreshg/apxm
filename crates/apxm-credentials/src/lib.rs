//! API credential management for LLM providers.
//!
//! Stores and retrieves credentials from `~/.apxm/credentials.toml` with
//! owner-only file permissions. Supports OpenAI, Anthropic, Google, and
//! Ollama provider backends.

pub mod credential;
pub mod mask;
pub mod validate;

use credential::{Credential, CredentialSummary, CredentialsFile};
use mask::mask_key;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use thiserror::Error;

const FILE_PERMISSIONS: u32 = 0o600;
const DIR_PERMISSIONS: u32 = 0o700;
const CREDENTIALS_FILENAME: &str = "credentials.toml";
const FILE_HEADER: &str = "# APXM Credentials - Managed by `apxm register`\n\
                            # Permissions: 0600 (owner read/write only)\n\
                            # DO NOT edit manually unless you know what you're doing.\n\n";

#[derive(Debug, Error)]
pub enum CredentialError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse credentials file: {0}")]
    Parse(#[from] toml::de::Error),

    #[error("Failed to serialize credentials: {0}")]
    Serialize(#[from] toml::ser::Error),

    #[error(
        "Credential '{name}' already exists. Remove it first with: apxm register remove {name}"
    )]
    AlreadyExists { name: String },

    #[error("Credential '{name}' not found")]
    NotFound { name: String },

    #[error("Unable to determine home directory")]
    HomeDirMissing,

    #[error(
        "Credential store at {path} has insecure permissions ({mode:o}). Fix with: chmod 600 {path}"
    )]
    InsecurePermissions { path: String, mode: u32 },

    #[error(
        "Credential store is inside a git repository at {repo_root}. Store credentials outside of git repos."
    )]
    InsideGitRepo { repo_root: String },

    #[error("Validation failed for '{name}': {reason}")]
    Validation { name: String, reason: String },
}

pub struct CredentialStore {
    path: PathBuf,
    dir: PathBuf,
}

impl CredentialStore {
    /// Open or create credential store at default path (~/.apxm/credentials.toml).
    pub fn open() -> Result<Self, CredentialError> {
        let home = dirs::home_dir().ok_or(CredentialError::HomeDirMissing)?;
        let dir = home.join(".apxm");
        let path = dir.join(CREDENTIALS_FILENAME);
        Ok(Self { path, dir })
    }

    /// Ensure the directory exists with correct permissions.
    fn ensure_dir(&self) -> Result<(), CredentialError> {
        if !self.dir.exists() {
            // Check if we're inside a git repo
            self.check_not_in_git_repo()?;
            fs::create_dir_all(&self.dir)?;
            fs::set_permissions(&self.dir, fs::Permissions::from_mode(DIR_PERMISSIONS))?;
        }
        // Create .gitignore as safety net
        let gitignore = self.dir.join(".gitignore");
        if !gitignore.exists() {
            fs::write(&gitignore, CREDENTIALS_FILENAME)?;
        }
        Ok(())
    }

    /// Check that the directory is not inside a git repository.
    fn check_not_in_git_repo(&self) -> Result<(), CredentialError> {
        let mut current = self.dir.as_path();
        loop {
            if current.join(".git").exists() {
                return Err(CredentialError::InsideGitRepo {
                    repo_root: current.display().to_string(),
                });
            }
            match current.parent() {
                Some(parent) => current = parent,
                None => break,
            }
        }
        Ok(())
    }

    /// Check file permissions are secure (owner-only read/write).
    fn check_permissions(&self) -> Result<(), CredentialError> {
        if !self.path.exists() {
            return Ok(());
        }
        let metadata = fs::metadata(&self.path)?;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(CredentialError::InsecurePermissions {
                path: self.path.display().to_string(),
                mode,
            });
        }
        Ok(())
    }

    /// Read the credentials file.
    fn read_file(&self) -> Result<CredentialsFile, CredentialError> {
        self.check_permissions()?;
        if !self.path.exists() {
            return Ok(CredentialsFile::default());
        }
        let contents = fs::read_to_string(&self.path)?;
        let file: CredentialsFile = toml::from_str(&contents)?;
        Ok(file)
    }

    /// Write the credentials file atomically.
    fn write_file(&self, file: &CredentialsFile) -> Result<(), CredentialError> {
        self.ensure_dir()?;
        let serialized = toml::to_string_pretty(file)?;
        let content = format!("{FILE_HEADER}{serialized}");

        // Atomic write via tempfile
        let temp = tempfile::NamedTempFile::new_in(&self.dir)?;
        fs::write(temp.path(), &content)?;
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(FILE_PERMISSIONS))?;
        temp.persist(&self.path)
            .map_err(std::io::Error::other)?;
        Ok(())
    }

    /// Add a credential. Fails if name already exists.
    pub fn add(&self, name: &str, cred: Credential) -> Result<(), CredentialError> {
        let mut file = self.read_file()?;
        if file.credentials.contains_key(name) {
            return Err(CredentialError::AlreadyExists {
                name: name.to_string(),
            });
        }
        file.credentials.insert(name.to_string(), cred);
        self.write_file(&file)
    }

    /// Remove a credential by name.
    pub fn remove(&self, name: &str) -> Result<(), CredentialError> {
        let mut file = self.read_file()?;
        if file.credentials.remove(name).is_none() {
            return Err(CredentialError::NotFound {
                name: name.to_string(),
            });
        }
        self.write_file(&file)
    }

    /// Get a credential by name.
    pub fn get(&self, name: &str) -> Result<Option<Credential>, CredentialError> {
        let file = self.read_file()?;
        Ok(file.credentials.get(name).cloned())
    }

    /// List all credentials with masked info.
    pub fn list(&self) -> Result<Vec<(String, CredentialSummary)>, CredentialError> {
        let file = self.read_file()?;
        Ok(file
            .credentials
            .into_iter()
            .map(|(name, cred)| {
                let summary = CredentialSummary {
                    provider: cred.provider.clone(),
                    masked_key: cred.api_key.as_deref().map(mask_key),
                    base_url: cred.base_url.clone(),
                    model: cred.model.clone(),
                    header_count: cred.headers.len(),
                };
                (name, summary)
            })
            .collect())
    }

    /// List all credentials with full data (for runtime use).
    pub fn list_all(&self) -> Result<Vec<(String, Credential)>, CredentialError> {
        let file = self.read_file()?;
        Ok(file.credentials.into_iter().collect())
    }

    /// Find credentials matching a provider name.
    pub fn find_by_provider(
        &self,
        provider: &str,
    ) -> Result<Vec<(String, Credential)>, CredentialError> {
        let file = self.read_file()?;
        Ok(file
            .credentials
            .into_iter()
            .filter(|(_, cred)| cred.provider == provider)
            .collect())
    }

    /// Generate config.toml entries from registered credentials.
    pub fn generate_config(&self) -> Result<String, CredentialError> {
        let file = self.read_file()?;
        let mut output = String::new();

        output.push_str("# Generated by `apxm register generate-config`\n\n");
        output.push_str("[chat]\nproviders = [");

        let names: Vec<&String> = file.credentials.keys().collect();
        for (i, name) in names.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(&format!("\"{}\"", name));
        }
        output.push_str("]\n\n");

        for (name, cred) in &file.credentials {
            output.push_str("[[llm_backends]]\n");
            output.push_str(&format!("name = \"{}\"\n", name));
            output.push_str(&format!("provider = \"{}\"\n", cred.provider));
            if let Some(key) = &cred.api_key {
                output.push_str(&format!("api_key = \"{}\"\n", key));
            }
            if let Some(model) = &cred.model {
                output.push_str(&format!("model = \"{}\"\n", model));
            }
            if let Some(endpoint) = &cred.base_url {
                output.push_str(&format!("endpoint = \"{}\"\n", endpoint));
            }
            if !cred.headers.is_empty() {
                output.push_str("\n[llm_backends.extra_headers]\n");
                for (k, v) in &cred.headers {
                    output.push_str(&format!("\"{}\" = \"{}\"\n", k, v));
                }
            }
            output.push('\n');
        }

        Ok(output)
    }

    /// Get the path to the credentials file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_store(dir: &Path) -> CredentialStore {
        let path = dir.join(CREDENTIALS_FILENAME);
        CredentialStore {
            path,
            dir: dir.to_path_buf(),
        }
    }

    #[test]
    fn add_and_get() {
        let tmp = TempDir::new().unwrap();
        let store = test_store(tmp.path());
        let cred = Credential {
            provider: "openai".to_string(),
            api_key: Some("sk-test-key-12345".to_string()),
            base_url: None,
            model: None,
            headers: Default::default(),
        };
        store.add("test", cred.clone()).unwrap();
        let got = store.get("test").unwrap().unwrap();
        assert_eq!(got.provider, "openai");
        assert_eq!(got.api_key.as_deref(), Some("sk-test-key-12345"));
    }

    #[test]
    fn add_duplicate_fails() {
        let tmp = TempDir::new().unwrap();
        let store = test_store(tmp.path());
        let cred = Credential {
            provider: "openai".to_string(),
            api_key: Some("sk-key".to_string()),
            base_url: None,
            model: None,
            headers: Default::default(),
        };
        store.add("dup", cred.clone()).unwrap();
        let result = store.add("dup", cred);
        assert!(matches!(result, Err(CredentialError::AlreadyExists { .. })));
    }

    #[test]
    fn remove_credential() {
        let tmp = TempDir::new().unwrap();
        let store = test_store(tmp.path());
        let cred = Credential {
            provider: "openai".to_string(),
            api_key: Some("sk-key".to_string()),
            base_url: None,
            model: None,
            headers: Default::default(),
        };
        store.add("rm-test", cred).unwrap();
        store.remove("rm-test").unwrap();
        assert!(store.get("rm-test").unwrap().is_none());
    }

    #[test]
    fn remove_nonexistent_fails() {
        let tmp = TempDir::new().unwrap();
        let store = test_store(tmp.path());
        let result = store.remove("nonexistent");
        assert!(matches!(result, Err(CredentialError::NotFound { .. })));
    }

    #[test]
    fn list_credentials() {
        let tmp = TempDir::new().unwrap();
        let store = test_store(tmp.path());
        store
            .add(
                "a",
                Credential {
                    provider: "openai".to_string(),
                    api_key: Some("sk-abcdefghijk".to_string()),
                    base_url: None,
                    model: None,
                    headers: Default::default(),
                },
            )
            .unwrap();
        store
            .add(
                "b",
                Credential {
                    provider: "anthropic".to_string(),
                    api_key: Some("sk-ant-test123456".to_string()),
                    base_url: None,
                    model: None,
                    headers: Default::default(),
                },
            )
            .unwrap();
        let list = store.list().unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn file_permissions() {
        let tmp = TempDir::new().unwrap();
        let store = test_store(tmp.path());
        store
            .add(
                "perm-test",
                Credential {
                    provider: "openai".to_string(),
                    api_key: Some("sk-key".to_string()),
                    base_url: None,
                    model: None,
                    headers: Default::default(),
                },
            )
            .unwrap();
        let metadata = fs::metadata(&store.path).unwrap();
        let mode = metadata.permissions().mode() & 0o777;
        assert_eq!(mode, FILE_PERMISSIONS);
    }

    #[test]
    fn generate_config_output() {
        let tmp = TempDir::new().unwrap();
        let store = test_store(tmp.path());
        store
            .add(
                "my-openai",
                Credential {
                    provider: "openai".to_string(),
                    api_key: Some("sk-test".to_string()),
                    base_url: None,
                    model: Some("gpt-4".to_string()),
                    headers: Default::default(),
                },
            )
            .unwrap();
        let config = store.generate_config().unwrap();
        assert!(config.contains("[[llm_backends]]"));
        assert!(config.contains("my-openai"));
        assert!(config.contains("gpt-4"));
    }

    #[test]
    fn headers_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let store = test_store(tmp.path());
        let mut headers = std::collections::BTreeMap::new();
        headers.insert("X-Custom".to_string(), "value".to_string());
        store
            .add(
                "with-headers",
                Credential {
                    provider: "openai".to_string(),
                    api_key: Some("sk-key".to_string()),
                    base_url: Some("https://example.com".to_string()),
                    model: None,
                    headers,
                },
            )
            .unwrap();
        let got = store.get("with-headers").unwrap().unwrap();
        assert_eq!(got.headers.get("X-Custom").unwrap(), "value");
    }
}
