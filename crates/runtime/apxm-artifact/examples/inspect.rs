use std::env;

use apxm_artifact::Artifact;

fn main() -> anyhow::Result<()> {
    let path = env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: inspect <artifact.apxmobj>"))?;

    let artifact = Artifact::read_from_path(&path)?;
    let metadata = artifact.metadata();
    let dag = artifact.dag();

    println!("Artifact: {}", path);
    println!(
        "Module: {}",
        metadata.module_name.as_deref().unwrap_or("<unknown>")
    );
    println!("Compiler version: {}", metadata.compiler_version);
    println!("Created at (ms): {}", metadata.created_at);
    println!("Nodes: {}", dag.nodes.len());
    println!("Edges: {}", dag.edges.len());
    println!("Entry nodes: {}", dag.entry_nodes.len());
    println!("Exit nodes: {}", dag.exit_nodes.len());

    Ok(())
}
