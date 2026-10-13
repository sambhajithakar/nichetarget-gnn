#!/usr/bin/env python3
"""Open Targets Platform GraphQL helper (stdlib only).

Usage:
  # target druggability for arbitrary gene symbols
  python3 fetch_ot.py targets ERBB2 ESR1 CXCL12 --out targets.csv
  python3 fetch_ot.py targets --genes-file genes.txt --out targets.csv
  # all target associations for a disease (EFO/MONDO id; obsolete EFO ids are
  # resolved by disease-name search, e.g. EFO_0000305 -> MONDO_0004989)
  python3 fetch_ot.py assoc EFO_0000305 --name "breast carcinoma" --out assoc.csv

Schema notes (API 26.9, data 26.09): `knownDrugs` was replaced by
`Target.drugAndClinicalCandidates { count rows { maxClinicalStage drug {id name} } }`;
tractability is `Target.tractability { label modality value }`.
"""
import argparse, csv, json, sys, time, urllib.request

API = "https://api.platform.opentargets.org/api/v4/graphql"


def gql(query, variables=None, retries=4):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            if out.get("errors"):
                raise RuntimeError(out["errors"])
            return out["data"]
        except Exception as e:  # noqa
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def resolve_symbols(symbols):
    """Map gene symbols -> Ensembl gene IDs (exact approvedSymbol match preferred)."""
    q = """query($s:String!){ search(queryString:$s, entityNames:["target"], page:{index:0,size:10}){
             hits { id object { ... on Target { approvedSymbol symbolSynonyms { label } } } } } }"""
    res = {}
    for s in symbols:
        hits = gql(q, {"s": s})["search"]["hits"]
        exact = [h for h in hits if h["object"] and h["object"]["approvedSymbol"].upper() == s.upper()]
        syn = [h for h in hits if h["object"] and s.upper() in
               [x["label"].upper() for x in (h["object"].get("symbolSynonyms") or [])]]
        pick = (exact or syn or [None])[0]
        res[s] = (pick["id"], pick["object"]["approvedSymbol"]) if pick else (None, None)
    return res


TARGET_Q = """query($id:String!){ target(ensemblId:$id){ id approvedSymbol approvedName
  tractability { label modality value }
  drugAndClinicalCandidates { count rows { maxClinicalStage drug { id name } } } } }"""

STAGE_ORDER = ["APPROVAL", "PHASE_4", "PHASE_3", "PHASE_2_3", "PHASE_2", "PHASE_1_2", "PHASE_1",
               "EARLY_PHASE_1", "PRECLINICAL", "UNKNOWN"]


def fetch_targets(symbols):
    ids = resolve_symbols(symbols)
    rows = []
    for s in symbols:
        eid, sym = ids[s]
        if not eid:
            rows.append({"query_symbol": s, "symbol": "", "ensembl_id": "", "note": "not found"})
            continue
        t = gql(TARGET_Q, {"id": eid})["target"]
        tr = t.get("tractability") or []
        pos = [x for x in tr if x["value"]]
        dc = t.get("drugAndClinicalCandidates") or {"count": 0, "rows": []}
        stages = [r["maxClinicalStage"] for r in dc["rows"]]
        drugs = sorted({r["drug"]["name"] for r in dc["rows"] if r.get("drug")})
        approved = sorted({r["drug"]["name"] for r in dc["rows"] if r.get("drug") and r["maxClinicalStage"] in ("APPROVAL", "PHASE_4")})
        best = min(stages, key=lambda x: STAGE_ORDER.index(x) if x in STAGE_ORDER else 99) if stages else ""
        rows.append({
            "query_symbol": s, "symbol": t["approvedSymbol"], "ensembl_id": t["id"],
            "n_drug_candidates": dc["count"], "n_unique_drugs": len(drugs), "max_clinical_stage": best,
            "approved_drugs": ";".join(approved), "all_drugs": ";".join(drugs),
            "tractable_SM": any(x["modality"] == "SM" for x in pos),
            "tractable_AB": any(x["modality"] == "AB" for x in pos),
            "tractable_PR": any(x["modality"] == "PR" for x in pos),
            "tractable_OC": any(x["modality"] == "OC" for x in pos),
            "tractability_labels_true": ";".join(f'{x["modality"]}:{x["label"]}' for x in pos),
        })
    return rows


def resolve_disease(did, name=None):
    q = "query($id:String!){ disease(efoId:$id){ id name } }"
    d = gql(q, {"id": did})["disease"]
    if d:
        return d["id"], d["name"]
    if not name:
        sys.exit(f"{did} not found in current release; pass --name to resolve by search")
    hits = gql("""query($s:String!){ search(queryString:$s, entityNames:["disease"], page:{index:0,size:5}){ hits { id name } } }""",
               {"s": name})["search"]["hits"]
    exact = [h for h in hits if h["name"].lower() == name.lower()] or hits
    print(f"note: {did} not in this release; using {exact[0]['id']} ({exact[0]['name']})", file=sys.stderr)
    return exact[0]["id"], exact[0]["name"]


def fetch_assoc(did, name=None, page_size=1000, max_rows=None):
    did, dname = resolve_disease(did, name)
    q = """query($id:String!,$i:Int!,$n:Int!){ disease(efoId:$id){ associatedTargets(page:{index:$i,size:$n}){
             count rows { score target { id approvedSymbol } } } } }"""
    rows, i = [], 0
    while True:
        at = gql(q, {"id": did, "i": i, "n": page_size})["disease"]["associatedTargets"]
        rows += [{"symbol": r["target"]["approvedSymbol"], "ensembl_id": r["target"]["id"], "score": r["score"]} for r in at["rows"]]
        i += 1
        if not at["rows"] or len(rows) >= at["count"] or (max_rows and len(rows) >= max_rows):
            break
    return did, dname, at["count"], rows


def write(rows, out):
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("targets"); a.add_argument("genes", nargs="*"); a.add_argument("--genes-file"); a.add_argument("--out", default="ot_targets.csv")
    b = sp.add_parser("assoc"); b.add_argument("disease_id"); b.add_argument("--name"); b.add_argument("--out", default="ot_assoc.csv")
    b.add_argument("--page-size", type=int, default=1000); b.add_argument("--max-rows", type=int)
    args = ap.parse_args()
    if args.cmd == "targets":
        genes = list(args.genes)
        if args.genes_file:
            genes += [l.strip() for l in open(args.genes_file) if l.strip()]
        rows = fetch_targets(genes)
        write(rows, args.out)
        for r in rows:
            print({k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v) for k, v in r.items()})
    else:
        did, dname, total, rows = fetch_assoc(args.disease_id, args.name, args.page_size, args.max_rows)
        write(rows, args.out)
        print(f"{did} ({dname}): {len(rows)} of {total} associated targets written to {args.out}")
