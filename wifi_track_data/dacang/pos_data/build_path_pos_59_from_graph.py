"""
从 graph(nodes.csv + edges.csv) + wifi_pos_new_59.csv + dacang_track_data_final_59.csv
生成 path_pos_59.csv（与 path_pos.csv 同结构：path + 列 1..N，单元格为 x:y）。

探针 -> 图节点：restored_x/y 与节点 (x,y) 最近邻；距离 > SNAP_BUFFER 仍采用最近邻并打印 WARN。
坐标输出保留 ROUND 位小数。

项目根目录运行:
  python wifi_track_data/dacang/pos_data/build_path_pos_59_from_graph.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DACANG = ROOT / "wifi_track_data" / "dacang"
F_GRAPH_NODES = DACANG / "pos_data" / "graph" / "nodes.csv"
F_GRAPH_EDGES = DACANG / "pos_data" / "graph" / "edges.csv"
F_WIFI = DACANG / "pos_data" / "wifi_pos_new_59.csv"
F_TRACK = DACANG / "track_data" / "dacang_track_data_final_59.csv"
F_OUT = DACANG / "pos_data" / "path_pos_59.csv"

ROUND = 3
SNAP_BUFFER = 0.01


def _norm_a_key(a) -> str:
    if isinstance(a, str) and a.isdigit():
        return str(int(a))
    try:
        f = float(a)
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(a)


def _xy_cell(x: float, y: float) -> str:
    return f"{round(float(x), ROUND):.{ROUND}f}:{round(float(y), ROUND):.{ROUND}f}"


def snap_wifi_to_node(nodes: pd.DataFrame, px: float, py: float) -> tuple[int, float]:
    best_i = int(nodes.iloc[0]["node_id"])
    best_d = math.inf
    for _, r in nodes.iterrows():
        d = math.hypot(float(r["x"]) - px, float(r["y"]) - py)
        if d < best_d:
            best_d = d
            best_i = int(r["node_id"])
    return best_i, best_d


def main() -> None:
    import sys

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    nodes = pd.read_csv(F_GRAPH_NODES)
    edges = pd.read_csv(F_GRAPH_EDGES)
    df_wifi = pd.read_csv(F_WIFI)
    df_track = pd.read_csv(F_TRACK)

    if not {"node_id", "x", "y"}.issubset(nodes.columns):
        raise ValueError("nodes.csv 需要列 node_id,x,y", nodes.columns.tolist())
    if not {"from_id", "to_id"}.issubset(edges.columns):
        raise ValueError("edges.csv 需要列 from_id,to_id", edges.columns.tolist())
    if not {"wifi", "restored_x", "restored_y"}.issubset(df_wifi.columns):
        raise ValueError("wifi_pos 需要列 wifi,restored_x,restored_y", df_wifi.columns.tolist())
    if not {"a", "t", "m"}.issubset(df_track.columns):
        raise ValueError("轨迹需要列 a,t,m", df_track.columns.tolist())

    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(int(r["from_id"]), int(r["to_id"]))

    wifi_to_node: dict[str, int] = {}
    snap_warn: list[tuple[str, float]] = []
    for _, r in df_wifi.iterrows():
        key = _norm_a_key(r["wifi"])
        nid, dist = snap_wifi_to_node(nodes, float(r["restored_x"]), float(r["restored_y"]))
        if dist > SNAP_BUFFER:
            snap_warn.append((key, dist))
        wifi_to_node[key] = nid

    # 轨迹中每一个探针 a 都应对应 wifi_pos 一行，且最近邻 node 须出现在 edges 构成的图中
    track_ids = {_norm_a_key(x) for x in df_track["a"].tolist()}
    not_in_wifi_pos = sorted(track_ids - set(wifi_to_node.keys()))

    print("--- 轨迹探针 与 graph 映射检查 ---")
    print(f"轨迹唯一探针 a 个数: {len(track_ids)}")
    if not_in_wifi_pos:
        print(
            "缺失: 轨迹中出现但 wifi_pos_new_59 无对应 wifi 行，无法映射到 graph 节点，探针 a:",
            not_in_wifi_pos,
            f"(共 {len(not_in_wifi_pos)} 个)",
        )
    else:
        print("OK: 全部唯一探针均在 wifi_pos_new_59 中有对应行。")

    outside_edge_graph: list[tuple[str, int]] = []
    for a in sorted(track_ids):
        if a not in wifi_to_node:
            continue
        nid = wifi_to_node[a]
        if nid not in G:
            outside_edge_graph.append((a, int(nid)))
    if outside_edge_graph:
        print(
            "缺失: 下列探针按坐标最近邻到的 node_id 未出现在任何边上（不属于当前 graph 顶点集），"
            "相关跨点最短路径将无法在图中计算:",
            outside_edge_graph,
            f"(共 {len(outside_edge_graph)} 个)",
        )
    else:
        print(
            "OK: 凡已在 wifi_pos 中的探针，其最近邻 node_id 均在 edges 构成的图中"
            f"（图顶点数 {G.number_of_nodes()}，边数 {G.number_of_edges()}）。",
        )

    df_track = df_track.copy()
    df_track["t"] = pd.to_datetime(df_track["t"])

    od_set: set[tuple[str, str]] = set()
    for _, g in df_track.groupby("m", sort=False):
        # 与轨迹分析 notebook 一致：同一时间戳多行须稳定次序，否则 od_set 会含虚假换探针对
        g = g.sort_values("t", kind="mergesort")
        aseq = g["a"].tolist()
        for i in range(1, len(aseq)):
            if aseq[i - 1] == aseq[i]:
                continue
            od_set.add((_norm_a_key(aseq[i - 1]), _norm_a_key(aseq[i])))

    rows: list[dict] = []
    missing_wifi: set[str] = set()
    no_path: list[tuple[str, str]] = []
    nodes_ix = nodes.set_index("node_id")

    for o, d in sorted(od_set, key=lambda x: (x[0], x[1])):
        if o not in wifi_to_node:
            missing_wifi.add(o)
            continue
        if d not in wifi_to_node:
            missing_wifi.add(d)
            continue
        u, v = wifi_to_node[o], wifi_to_node[d]
        if u == v:
            row = nodes_ix.loc[u]
            cells = [_xy_cell(row["x"], row["y"])]
        else:
            try:
                sp = nx.shortest_path(G, u, v)
            except nx.NetworkXNoPath:
                no_path.append((o, d))
                continue
            cells = []
            for nid in sp:
                row = nodes_ix.loc[nid]
                cells.append(_xy_cell(row["x"], row["y"]))
        rows.append({"path": f"{o}->{d}", "_cells": cells})

    if missing_wifi:
        s = sorted(missing_wifi)
        print(
            "WARN: 以下 a 不在 wifi_pos_new_59 中，已跳过相关 OD:",
            s[:40],
            f"... total {len(missing_wifi)}",
        )
    if snap_warn:
        far = sorted(snap_warn, key=lambda x: -x[1])[:15]
        print(
            f"WARN: {len(snap_warn)} 个探针最近节点距离 > {SNAP_BUFFER}（仍用最近邻）, 例如:",
            far,
        )
    if no_path:
        print(f"WARN: 图中无路，跳过 {len(no_path)} 对 OD，例如:", no_path[:20])

    if not rows:
        raise RuntimeError("没有可写出的路径行，请检查数据与图连通性")

    max_k = max(len(r["_cells"]) for r in rows)
    max_k = max(max_k, 64)
    cols = ["path"] + [str(i) for i in range(1, max_k + 1)]

    out_rows = []
    for r in rows:
        dct = {"path": r["path"]}
        for i in range(1, max_k + 1):
            dct[str(i)] = r["_cells"][i - 1] if i - 1 < len(r["_cells"]) else float("nan")
        out_rows.append(dct)

    out_df = pd.DataFrame(out_rows, columns=cols)
    out_df.to_csv(F_OUT, index=False, encoding="utf-8-sig")
    print("Wrote", F_OUT, "rows:", len(out_df), "max vertex columns:", max_k)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
