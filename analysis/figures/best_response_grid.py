from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
HP = 380
CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    #"DPR_TABLE_140_UPDATED - HP_140_DPR_Profiles__Updated_.csv",
    f"/Users/gabesmithline/Desktop/SRG/melo_project/analysis/figures/dpr_table_hp_{HP}.csv"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "best_response_grid.png")

# Deviations configuration ---------------------------------------------------
DEV_TYPES = {
    "mobi_melo_to_cda": {
        "color": "#B8860B",  # maize-like ##FFCC66
        "pre_cond": lambda p: p["#_MOBI_MELO"] > 0,
        "profile_change": lambda p: (
            p["#_MOBI_MELO"] - 1,
            p["#_MOBI_CDA"] + 1,
            p["#_ZI_MELO"],
            p["#_ZI_CDA"],
        ),
        "dev_type": "mobi_melo_to_cda",
        "payoff_from": "Pay_MOBI_MELO",
        "payoff_to": "Pay_MOBI_CDA",
    },
    "mobi_cda_to_melo": {
        "color": "#B8860B", ##FFCC66
        "pre_cond": lambda p: p["#_MOBI_CDA"] > 0,
        "profile_change": lambda p: (
            p["#_MOBI_MELO"] + 1,
            p["#_MOBI_CDA"] - 1,
            p["#_ZI_MELO"],
            p["#_ZI_CDA"],
        ),
        "dev_type": "mobi_cda_to_melo",
        "payoff_from": "Pay_MOBI_CDA",
        "payoff_to": "Pay_MOBI_MELO",
    },
    "zi_melo_to_cda": {
        "color": "royalblue",
        "pre_cond": lambda p: p["#_ZI_MELO"] > 0,
        "profile_change": lambda p: (
            p["#_MOBI_MELO"],
            p["#_MOBI_CDA"],
            p["#_ZI_MELO"] - 1,
            p["#_ZI_CDA"] + 1,
        ),
        "dev_type": "zi_melo_to_cda",
        "payoff_from": "Pay_ZI_MELO",
        "payoff_to": "Pay_ZI_CDA",
    },
    "zi_cda_to_melo": {
        "color": "royalblue",
        "pre_cond": lambda p: p["#_ZI_CDA"] > 0,
        "profile_change": lambda p: (
            p["#_MOBI_MELO"],
            p["#_MOBI_CDA"],
            p["#_ZI_MELO"] + 1,
            p["#_ZI_CDA"] - 1,
        ),
        "dev_type": "zi_cda_to_melo",
        "payoff_from": "Pay_ZI_CDA",
        "payoff_to": "Pay_ZI_MELO",
    },
}

def read_dpr_table(csv_path: str) -> pd.DataFrame:
    """Load the DPR CSV and keep only the first 8 columns (ignore extra notes)."""
    df = pd.read_csv(csv_path)
    # Keep only the first 8 data columns in case there is a trailing comment column
    df = df.iloc[:, :8]

    # Explicitly convert every column to numeric and alert if any coercions happen
    for col in df.columns:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_na = df[col].isna().sum()
        if after_na > before_na:
            print(f"[WARN] Column '{col}' had {after_na-before_na} non-numeric entries; coerced to NaN.")

    # Drop rows with NaNs – the table is supposed to be complete
    if df.isna().any().any():
        raise ValueError("NaNs detected after numeric conversion; please verify the CSV integrity.")

    return df

def build_profile_lookup(df: pd.DataFrame):
    """Return a dict mapping profile tuple to row index for O(1) lookup."""
    lookup = {}
    for idx, row in df.iterrows():
        profile = (
            int(row["#_MOBI_MELO"]),
            int(row["#_MOBI_CDA"]),
            int(row["#_ZI_MELO"]),
            int(row["#_ZI_CDA"]),
        )
        lookup[profile] = idx
    return lookup


def print_basins_of_attraction(G):
    """Print directed basins of attraction for each sink equilibrium in G.

    A sink equilibrium e is a node with out-degree zero. Its basin is
    all nodes that can reach e via a directed path (including e itself).
    """
    # 1. Identify equilibria (sink nodes)
    equilibria = [node for node in G.nodes() if G.out_degree(node) == 0]

    # 2. Compute directed basins via ancestors
    basins = {}
    for e in equilibria:
        # ancestors(G, e) returns all v such that there is a path v -> ... -> e
        basin_nodes = set(nx.ancestors(G, e))
        basin_nodes.add(e)
        basins[e] = sorted(basin_nodes)

    # 3. Print results
    print(f"\nBasins of attraction for HP {HP}:")
    for e, members in basins.items():
        print(f"Equilibrium {e}: basin = {members}")
    
    

    src = (2,2,3,1)
    dst = (0,4,0,4)

    # 1. Check Boolean reachability
    print("Has path?", nx.has_path(G, src, dst))

    # 2. If True, list one of the possible paths
    if nx.has_path(G, src, dst):
        path = next(nx.all_simple_paths(G, src, dst))
        print("Example path:", path)
    else:
        print("No directed path exists from", src, "to", dst)


def main():
    df = read_dpr_table(CSV_PATH)
    lookup = build_profile_lookup(df)

    G = nx.MultiDiGraph()
    pos = {}

    # 1) Add nodes with fixed lattice positions
    for idx, row in df.iterrows():
        profile = (
            int(row["#_MOBI_MELO"]),
            int(row["#_MOBI_CDA"]),
            int(row["#_ZI_MELO"]),
            int(row["#_ZI_CDA"]),
        )
        G.add_node(profile)
        # x = MOBI-MELO, y = ZI-MELO
        pos[profile] = (profile[0], profile[2])

    # 2) Add edges for each deviation type
    for idx, row in df.iterrows():
        profile = (
            int(row["#_MOBI_MELO"]),
            int(row["#_MOBI_CDA"]),
            int(row["#_ZI_MELO"]),
            int(row["#_ZI_CDA"]),
        )

        for dev_name, dev in DEV_TYPES.items():
            if not dev["pre_cond"](row):
                continue  # no such agent to deviate
            target_profile = dev["profile_change"](row)
            if target_profile not in lookup:
                # Deviation profile missing from table → skip
                continue

            from_payoff = float(row[dev["payoff_from"]])
            to_payoff = float(df.loc[lookup[target_profile], dev["payoff_to"]])

            edge_color = dev["color"]
            # Log details of the comparison for full transparency
            print(
                f"DEV {dev_name:15} | {profile} -> {target_profile} | "
                f"from_payoff={from_payoff} | to_payoff={to_payoff}"
            )

            if to_payoff > from_payoff:
                print("    → Profitable deviation – adding directed edge")
                G.add_edge(profile, target_profile, key=dev_name, color=edge_color, dev_type=dev_name)
            

    # Print all edges for verification
    print("\nFinal graph edges:")
    for u, v, data in G.edges(data=True):
        
        print(f"{u} -> {v} | color={data['color']}")

    # 3) Plot -----------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    # Draw nodes (cyan with navy edges)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2000, #1200
        node_color="cyan",
        edgecolors="royalblue",
        linewidths=1.5,
        ax=ax,
    )

    # Node labels as tuple
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: f"{n[0]},{n[1]}\n{n[2]},{n[3]}" for n in G.nodes()},
        font_size=12, # 10
        font_color="black",
        font_weight="bold",
        ax=ax,
    )

    # Draw edges grouped by colour, distinguishing loops vs arrows
    loop_rad_map = {
        # "black": 0.25,
        # "grey": 0.35,
        # "navy": 0.45,
        # "#FFCC66": 0.55,  # maize
        "mobi_cda_to_melo": 0.35,
        "zi_melo_to_cda": 0.25,
        "zi_cda_to_melo": 0.45,
        "#FFCC66": 0.55,
    }

    # Organise edges by colour
    edges_by_dev_type = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        # Group edges by deviation type while keeping colour
        dev_type = data["dev_type"]
        edges_by_dev_type.setdefault(dev_type, {"color": data["color"], "edges": []})["edges"].append((u, v))

    for dev_type, info in edges_by_dev_type.items():
        color = info["color"]
        edgelist = info["edges"]
        if not edgelist:
            continue
        # Separate self-loops vs regular edges so we can use different rad values
        loop_edges = [(u, v) for u, v in edgelist if u == v]
        normal_edges = [(u, v) for u, v in edgelist if u != v]

        if normal_edges:
            base_rad = {
                "mobi_cda_to_melo": 0.35,
                "zi_melo_to_cda": 0.25,
                "zi_cda_to_melo": 0.45,
                "#mobi_melo_to_cda": 0.55,
            }.get(dev_type, 0.3)
            # Draw each edge separately so opposite directions get opposite curvature
            for u, v in normal_edges:
                sign = 1 if str(u) < str(v) else -1  # deterministic ordering
                rad = sign * base_rad
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(u, v)],
                    edge_color=color,
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=56, #24
                    width=2.8, #2.2
                    min_source_margin=8, #5
                    min_target_margin=8, #5
                    connectionstyle=f"arc3,rad={rad}",
                    ax=ax,
                )
        if loop_edges:
            rad = loop_rad_map.get(dev_type, 0.3)
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=loop_edges,
                edge_color=color,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=56, #24 
                width=2.8, #2.2
                min_source_margin=8, #5
                min_target_margin=8, #5
                connectionstyle=f"arc3,rad={rad}",
                ax=ax,
            )

    # 4) Set axis properties after drawing
    ax.set_xlabel("# MOBIs in M-ELO", fontweight="bold", fontsize=14)
    ax.set_ylabel("# Background Traders in M-ELO", fontweight="bold", fontsize=14)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([str(i) for i in range(5)], fontweight="bold", fontsize=14)
    ax.set_yticklabels([str(i) for i in range(5)], fontweight="bold", fontsize=14)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.tick_params(axis='x', which='major', labelsize=14, pad=8)
    ax.tick_params(axis='y', which='major', labelsize=14, pad=8)
    ax.set_frame_on(True)
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)

    # Legend
    legend_handles = [
        mpatches.FancyArrowPatch((0, 0), (0.3, 0), color=dev["color"], label=label)
        for label, dev in zip(
            [
                "MOBI-MELO → CDA",
                "MOBI-CDA  → MELO",
                "ZI-MELO   → CDA",
                "ZI-CDA    → MELO",
            ],
            DEV_TYPES.values(),
        )
    ]
    # ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Best-response grid saved to {OUTPUT_PATH}")
    # Analyse basins of attraction
    print_basins_of_attraction(G)

if __name__ == "__main__":
    main()