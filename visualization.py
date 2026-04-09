import os
import networkx as nx
from pyvis.network import Network
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationEngine:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.pdf_dir = os.path.join(self.output_dir, "paper_summaries")
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)

    def generate_paper_pdf(self, paper_data, graph_metrics):
        """
        Generates a detailed PDF summary for an individual paper.
        """
        doi_safe = paper_data.get("doi", "unknown").replace("/", "_").replace(":", "_")
        file_path = os.path.join(self.pdf_dir, f"{doi_safe}_summary.pdf")
        
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title and Metadata
        story.append(Paragraph(f"Paper Summary: {paper_data.get('title', 'Unknown Title')}", styles['Title']))
        story.append(Spacer(1, 12))
        
        metadata_style = ParagraphStyle('Metadata', parent=styles['Normal'], fontSize=10, spaceAfter=6)
        story.append(Paragraph(f"<b>Authors:</b> {', '.join(paper_data.get('authors', ['Unknown']))}", metadata_style))
        story.append(Paragraph(f"<b>Year:</b> {paper_data.get('year', 'N/A')}", metadata_style))
        story.append(Paragraph(f"<b>DOI:</b> {paper_data.get('doi', 'N/A')}", metadata_style))
        story.append(Spacer(1, 12))

        # Summary/Abstract
        story.append(Paragraph("Abstract Summary", styles['Heading2']))
        story.append(Paragraph(paper_data.get('abstract', 'No abstract available.'), styles['Normal']))
        story.append(Spacer(1, 12))

        # Graph Insights
        story.append(Paragraph("Knowledge Graph Insights", styles['Heading2']))
        story.append(Paragraph(f"<b>Cluster ID:</b> {paper_data.get('cluster_id', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Centrality Score:</b> {graph_metrics.get('centrality', 0):.4f}", styles['Normal']))
        
        role = "Key Relevant Paper" if graph_metrics.get('centrality', 0) > 0.5 else "Supporting Evidence"
        if paper_data.get('gap_score', 0) > 0.7:
            role = "Research Gap / Frontier Paper"
            
        story.append(Paragraph(f"<b>Role in Landscape:</b> {role}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Related Papers
        story.append(Paragraph("Top Related Papers in Graph", styles['Heading3']))
        for related in graph_metrics.get('top_related', []):
            story.append(Paragraph(f"- {related}", styles['Normal']))

        doc.build(story)
        return file_path

    def create_interactive_graph(self, input_data, filename="knowledge_graph.html"):
        """
        Creates a PyVis interactive graph based on the input structure.
        """
        G = nx.Graph()
        
        nodes = input_data.get("graph_nodes", [])
        edges = input_data.get("graph_edges", [])
        metrics = input_data.get("networkx_metrics", {})
        centrality = metrics.get("centrality_scores", {})
        gap_scores = metrics.get("gap_scores", {})
        rejected = input_data.get("rejected_papers", [])

        # Add Nodes
        for node in nodes:
            node_id = node.get("doi") or node.get("title")
            
            # Category and Color logic
            color = "#3498db" # Blue (Similar/Clustered)
            node_centrality = centrality.get(node_id, 0)
            node_gap = gap_scores.get(node_id, 0)
            
            if node_centrality > 0.2: # High Centrality threshold
                color = "#2ecc71" # Green (Relevant)
            if node_gap > 0.7: # High Gap score threshold
                color = "#e74c3c" # Red (Gap/Frontier)
                
            # Sizing based on centrality
            size = 15 + (node_centrality * 50)
            
            # Tooltip
            tooltip = f"""
            <b>Title:</b> {node.get('title')}<br>
            <b>Year:</b> {node.get('year')}<br>
            <b>DOI:</b> {node.get('doi')}<br>
            <b>Cluster:</b> {node.get('cluster_id')}<br>
            <b>Score:</b> {node.get('score', 0):.2f}<br>
            <b>Gap Score:</b> {node_gap:.2f}
            """
            
            G.add_node(node_id, label=node.get('title')[:30]+"...", title=tooltip, color=color, size=size)

        # Add Edges
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            weight = edge.get("weight", 1.0)
            if G.has_node(source) and G.has_node(target):
                G.add_edge(source, target, value=weight*5, title=f"Similarity: {weight:.2f}")

        # Render PyVis
        net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", directed=False)
        net.from_nx(G)
        
        # Configure Physics for ForceAtlas2-like behavior
        net.force_atlas_2based()
        
        # Custom HTML Injection for Legend and Synthesis Panel
        final_report = input_data.get("final_report", {})
        legend_html = f"""
        <div id="graph-ui" style="position: absolute; top: 10px; right: 10px; width: 300px; background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #ddd; z-index: 1000; font-family: sans-serif; height: 90%; overflow-y: auto;">
            <h3>Research Insights</h3>
            <div style="background: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p><b>Themes:</b> {final_report.get('trends', 'N/A')}</p>
                <p><b>Gaps:</b> {final_report.get('gaps', 'N/A')}</p>
                <p><b>Contradictions:</b> {final_report.get('contradictions', 'N/A')}</p>
            </div>
            
            <h4>Legend</h4>
            <p><span style='background:#2ecc71; padding:3px 8px; border-radius:10px;'>&nbsp;</span> Most Relevant Papers</p>
            <p><span style='background:#e74c3c; padding:3px 8px; border-radius:10px;'>&nbsp;</span> Research Gap / Frontier</p>
            <p><span style='background:#3498db; padding:3px 8px; border-radius:10px;'>&nbsp;</span> Clustered Themes</p>
            <p><span style='background:#95a5a6; padding:3px 8px; border-radius:10px;'>&nbsp;</span> Rejected Papers</p>
            
            <h4>Rejected Analysis ({len(rejected)})</h4>
            <div style="font-size: 11px;">
        """
        for r in rejected:
            legend_html += f"<p><b>{r.get('title')[:40]}...</b><br><i style='color:#7f8c8d;'>Reason: {r.get('reason')}</i></p>"
            
        legend_html += "</div></div>"
        
        # Save HTML and Patch with Manual Legend
        out_path = os.path.join(self.output_dir, filename)
        net.save_graph(out_path)
        
        with open(out_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Insert our custom UI before the closing body tag
        patched_html = html_content.replace("</body>", f"{legend_html}</body>")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(patched_html)
            
        return out_path

# MOCK DATA FOR STANDALONE DEMO
if __name__ == "__main__":
    mock_input = {
        "graph_nodes": [
            {"title": "Neural Radiance Fields in Medical Imaging", "year": 2023, "doi": "10.123/med.1", "cluster_id": 1, "score": 0.85, "abstract": "This paper presents a new NeRF architecture for CT reconstructon with high fidelity."},
            {"title": "Efficient NeRF Sampling", "year": 2022, "doi": "10.123/eff.1", "cluster_id": 1, "score": 0.78, "abstract": "A study on importance sampling in volumetric rendering with radiance fields."},
            {"title": "Unsolved Problems in Volumetric Rendering", "year": 2024, "doi": "10.123/gap.1", "cluster_id": 2, "score": 0.92, "abstract": "Identification of edge cases where NeRF fails to represent transparent surfaces."}
        ],
        "graph_edges": [
            {"source": "10.123/med.1", "target": "10.123/eff.1", "weight": 0.65},
            {"source": "10.123/med.1", "target": "10.123/gap.1", "weight": 0.20}
        ],
        "networkx_metrics": {
            "centrality_scores": {"10.123/med.1": 0.8, "10.123/eff.1": 0.4, "10.123/gap.1": 0.3},
            "gap_scores": {"10.123/gap.1": 0.95, "10.123/med.1": 0.1}
        },
        "rejected_papers": [
            {"title": "Generic Radiance Study", "reason": "Too similar to already selected papers (MMR redundancy)"},
            {"title": "Intro to Graphics 101", "reason": "Low semantic relevance to query"}
        ],
        "final_report": {
            "trends": "Increasing focus on medical applications of NeRF.",
            "gaps": "Surface transparency modeling remains unsolved.",
            "contradictions": "Sampling efficiency vs reconstruction quality trade-offs."
        }
    }

    viz = VisualizationEngine()
    
    # Generate a sample PDF for one paper
    viz.generate_paper_pdf(mock_input["graph_nodes"][0], {"centrality": 0.8, "top_related": ["Efficient NeRF Sampling"]})
    
    # Generate Graph
    graph_html = viz.create_interactive_graph(mock_input)
    print(f"Visualization complete. Files generated in the 'outputs' directory.")
    print(f"Graph Path: {graph_html}")
