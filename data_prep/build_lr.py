"""Build combined ligand-receptor pair tables from offline resources.
human: NicheNet v2 (OmniPath+Verschueren), CellPhoneDB v2.1.7 (complexes split into subunits),
       NicheNet v1 curated subsets (ramilowski, kegg, guide2pharmacology; ppi_prediction excluded)
mouse: NicheNet v2 mouse."""
import pandas as pd
B='/home/user/data/lr/'
rows=[]
h=pd.read_csv(B+'nichenet_v2/lr_network_human_21122021.csv')
rows.append(pd.DataFrame({'ligand':h['from'],'receptor':h['to'],'source':'nichenet_v2:'+h['database']}))
v1=pd.read_csv(B+'nichenet_v1/lr_network.txt',sep='\t'); v1=v1[~v1.database.str.startswith('ppi_prediction')]
rows.append(pd.DataFrame({'ligand':v1['from'],'receptor':v1['to'],'source':'nichenet_v1:'+v1['database']}))
# CellPhoneDB
D=B+'cellphonedb_v2.1.7/data/'
prot=pd.read_csv(D+'protein_input.csv'); gene=pd.read_csv(D+'gene_input.csv'); cx=pd.read_csv(D+'complex_input.csv'); it=pd.read_csv(D+'interaction_input.csv')
u2g=gene.drop_duplicates('uniprot').set_index('uniprot')['hgnc_symbol'].to_dict()
isrec={u:str(r).lower()=='true' for u,r in zip(prot.uniprot,prot.receptor)}
members={}
for _,r in cx.iterrows():
    members[r.complex_name]=[r[c] for c in ['uniprot_1','uniprot_2','uniprot_3','uniprot_4'] if isinstance(r[c],str)]
    isrec[r.complex_name]=str(r.receptor).lower()=='true'
def genes(p): return [u2g.get(u) for u in members.get(p,[p]) if u2g.get(u)]
n_swap=0
for _,r in it.iterrows():
    a,b=r.partner_a,r.partner_b
    if isrec.get(a,False) and not isrec.get(b,False): a,b=b,a; n_swap+=1
    for ga in genes(a):
        for gb in genes(b):
            rows.append(pd.DataFrame({'ligand':[ga],'receptor':[gb],'source':['cellphonedb_v2.1.7']}))
out=pd.concat(rows).dropna().drop_duplicates()
out=out[out.ligand!=out.receptor]
out.to_csv(B+'lr_pairs_human.csv',index=False)
print('human rows',len(out),'unique pairs',len(out[['ligand','receptor']].drop_duplicates()),'cpdb swapped',n_swap)
print(out.groupby(out.source.str.split(':').str[0]).size())
m=pd.read_csv(B+'nichenet_v2/lr_network_mouse_21122021.csv')
mo=pd.DataFrame({'ligand':m['from'],'receptor':m['to'],'source':'nichenet_v2:'+m['database']}).dropna().drop_duplicates()
mo.to_csv(B+'lr_pairs_mouse.csv',index=False); print('mouse rows',len(mo),'unique pairs',len(mo[['ligand','receptor']].drop_duplicates()))
