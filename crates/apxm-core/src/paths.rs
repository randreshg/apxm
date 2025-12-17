//! Helpers for resolving APXM workspace directories.
//!
//! Mirrors Codex-style layout:
//! - Global home at `$APXM_HOME` or `~/.apxm`
//! - Project-scoped folder at `<repo>/.apxm`
//!   - `artifacts/`, `cache/`, `logs/` are created lazily

use dirs::home_dir;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const ENV_HOME: &str = "APXM_HOME";
const PROJECT_DIR: &str = ".apxm";
const ARTIFACTS_DIR: &str = "artifacts";
const CACHE_DIR: &str = "cache";
const LOGS_DIR: &str = "logs";

/// Resolved APXM directories for the current process.
#[derive(Debug, Clone)]
pub struct ApxmPaths {
    home_dir: PathBuf,
    project_dir: PathBuf,
}

impl ApxmPaths {
    /// Discover paths based on current working directory and environment.
    pub fn discover() -> io::Result<Self> {
        let cwd = env::current_dir()?;

        let project_candidate = Self::resolve_project_dir(&cwd);
        fs::create_dir_all(&project_candidate)?;
        let project_dir = project_candidate
            .canonicalize()
            .unwrap_or(project_candidate);

        let home_dir = match Self::resolve_home_dir() {
            Ok(home) => match fs::create_dir_all(&home) {
                Ok(_) => home.canonicalize().unwrap_or(home),
                Err(_) => project_dir.clone(),
            },
            Err(_) => project_dir.clone(),
        };

        Ok(Self {
            home_dir,
            project_dir,
        })
    }

    fn resolve_home_dir() -> io::Result<PathBuf> {
        if let Ok(path) = env::var(ENV_HOME) {
            return Ok(PathBuf::from(path));
        }

        let home = home_dir().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "Unable to determine user home directory",
            )
        })?;
        Ok(home.join(PROJECT_DIR))
    }

    fn resolve_project_dir(start: &Path) -> PathBuf {
        for ancestor in start.ancestors() {
            let candidate = ancestor.join(PROJECT_DIR);
            if candidate.is_dir() {
                return candidate;
            }
        }
        start.join(PROJECT_DIR)
    }

    /// Global home directory (typically `~/.apxm`).
    pub fn home_dir(&self) -> &Path {
        &self.home_dir
    }

    /// Project-specific directory (`<repo>/.apxm`).
    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    /// Path to the project-scoped configuration file.
    pub fn project_config_path(&self) -> PathBuf {
        self.project_dir.join("config.toml")
    }

    fn ensure_subdir(&self, name: &str) -> io::Result<PathBuf> {
        let path = self.project_dir.join(name);
        fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Directory for compiled artifacts, e.g. `.apxm/artifacts`.
    pub fn artifacts_dir(&self) -> io::Result<PathBuf> {
        self.ensure_subdir(ARTIFACTS_DIR)
    }

    /// Directory for cached files, e.g. `.apxm/cache`.
    pub fn cache_dir(&self) -> io::Result<PathBuf> {
        self.ensure_subdir(CACHE_DIR)
    }

    /// Directory for logs, e.g. `.apxm/logs`.
    pub fn logs_dir(&self) -> io::Result<PathBuf> {
        self.ensure_subdir(LOGS_DIR)
    }
}
