
import streamlit as st
from groq import Groq
import os, json, re, math
from datetime import date

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Dreamhouse AI", page_icon="🏡")
st.markdown("""<style>
  #MainMenu,footer{visibility:hidden;}
  .block-container{padding:1rem 1.5rem 0 1.5rem;}
  html,body,[data-testid="stAppViewContainer"]{background:#060d1f!important;}
  [data-testid="stSidebar"]{background:#0d1b35!important;}
  .ub{background:linear-gradient(135deg,#1a3a8a,#2a5ac0);
      border:1px solid rgba(74,158,255,.35);border-radius:16px 16px 4px 16px;
      padding:10px 16px;margin:6px 0 6px 15%;
      color:rgba(230,240,255,.95);font-size:.88rem;line-height:1.6;}
  .ab{background:rgba(255,255,255,.05);border:1px solid rgba(74,158,255,.15);
      border-radius:4px 16px 16px 16px;padding:10px 16px;margin:6px 15% 6px 0;
      color:rgba(220,235,255,.9);font-size:.88rem;line-height:1.6;}
  .stChatInput textarea{background:rgba(255,255,255,.06)!important;
      color:rgba(220,235,255,.95)!important;}
</style>""", unsafe_allow_html=True)


SYSTEM_PROMPT = """You are Dreamhouse AI, an expert architectural consultant. Collect home requirements through warm, enthusiastic conversation, then output a blueprint JSON.

COLLECT ALL (ask 2-3 per message, never more):
1.  Plot SHAPE: rectangular / L-shaped / trapezoidal / triangular / corner
2.  Plot dimensions (per shape guide below)
3.  Number of floors
4.  Total house area (sq ft)
5.  Bedrooms: count + sizes (master, standard)
6.  Bathrooms: ensuite attached to master + shared full baths
7.  Living room: open-plan or separate
8.  Kitchen: open/closed, island or not
9.  Dining: separate room or combined with kitchen/living
10. Garage: yes/no, how many cars
11. Extra rooms: study, laundry
12. Special: foyer/entry hall, staircase (if multi-floor), balcony
13. Style: modern / traditional / farmhouse / mediterranean / contemporary / craftsman

PLOT SHAPE GUIDE (Neufert Architects Data, 6th ed.):
- rectangular:  plotWidth × plotLength  (most common)
- trapezoidal:  frontWidth (street side) + backWidth + plotLength
- L-shaped:     plotWidth × plotLength + notchWidth × notchLength at notchCorner
                notchCorner = top-right / top-left / bottom-right / bottom-left
- triangular:   plotBase (wide street end) × plotHeight  (pie/fan lots)
- corner:       plotWidth × plotLength + cornerCut in feet (diagonal setback)

When you have collected EVERYTHING, output EXACTLY the word "BLUEPRINT_READY" on its own line, then immediately the JSON block:
```json
{
  "plotShape":"rectangular","plotWidth":60,"plotLength":80,
  "floors":2,"houseArea":2800,"style":"modern",
  "rooms":[
    {"type":"garage",   "name":"Garage",        "width":20,"length":22,"cars":2},
    {"type":"entry",    "name":"Entry Hall",     "width":10,"length":10},
    {"type":"study",    "name":"Study",          "width":11,"length":12},
    {"type":"living",   "name":"Living Room",    "width":18,"length":20},
    {"type":"kitchen",  "name":"Kitchen",        "width":14,"length":16,"hasIsland":true},
    {"type":"dining",   "name":"Dining Room",    "width":12,"length":14},
    {"type":"staircase","name":"Staircase",      "width":10,"length":12},
    {"type":"bedroom",  "name":"Master Bedroom", "width":16,"length":18,"isMaster":true},
    {"type":"bathroom", "name":"Master Bath",    "width":9, "length":11,"isEnsuite":true},
    {"type":"bedroom",  "name":"Bedroom 2",      "width":12,"length":13},
    {"type":"bedroom",  "name":"Bedroom 3",      "width":12,"length":13},
    {"type":"bathroom", "name":"Shared Bath",    "width":8, "length":10},
    {"type":"laundry",  "name":"Laundry",        "width":7, "length":9}
  ]
}
```
Shape extras — trapezoidal: add "frontWidth":50,"backWidth":70
L-shaped: add "notchWidth":20,"notchLength":25,"notchCorner":"top-right"
triangular: add "plotBase":60,"plotHeight":80
corner: add "cornerCut":12
IRC minimums: bedroom ≥ 70 sf (min 7 ft dim), all habitable ≥ 70 sf, hallway ≥ 3 ft wide."""

RC = {
    "garage":   {"f":"rgba(148,155,182,.48)","s":"#5566aa","w":"#2a3355","a":"#aab0cc"},
    "entry":    {"f":"rgba(255,210,70,.48)", "s":"#cc9900","w":"#775500","a":"#ffe080"},
    "study":    {"f":"rgba(185,140,255,.48)","s":"#8833ee","w":"#551199","a":"#cc99ff"},
    "living":   {"f":"rgba(255,88,88,.48)",  "s":"#cc1111","w":"#880000","a":"#ff9999"},
    "kitchen":  {"f":"rgba(255,195,50,.48)", "s":"#aa6600","w":"#663300","a":"#ffdd88"},
    "dining":   {"f":"rgba(255,160,60,.48)", "s":"#994400","w":"#662200","a":"#ffbb77"},
    "staircase":{"f":"rgba(200,175,140,.48)","s":"#886633","w":"#553311","a":"#ccaa77"},
    "bedroom":  {"f":"rgba(50,138,255,.48)", "s":"#0044bb","w":"#002266","a":"#88bbff"},
    "bathroom": {"f":"rgba(48,198,135,.48)", "s":"#007733","w":"#004422","a":"#66ddaa"},
    "laundry":  {"f":"rgba(88,208,208,.48)", "s":"#006666","w":"#003333","a":"#88eeee"},
    "hallway":  {"f":"rgba(225,225,190,.16)","s":"#555533","w":"#333322","a":"#aaaa88"},
    "courtyard":{"f":"rgba(22,100,40,.28)",  "s":"#155528","w":"#0a2a14","a":"#44bb66"},
    "default":  {"f":"rgba(180,180,180,.32)","s":"#666666","w":"#333333","a":"#999999"},
}


def plot_geo(spec):
    """
    Returns:
      pw, pl   — bounding box (feet)
      uw, uh   — usable interior (minus wall margin M)
      ox, oy   — interior origin (= M, M)
      poly     — list of (x,y) feet for plot boundary polygon
      shape    — normalised shape string
      extras   — shape-specific measurements for annotation
    """
    M = 2.0  
    sh = spec.get("plotShape","rectangular").lower().replace("-","").replace(" ","")
    pw = float(spec.get("plotWidth", 60))
    pl = float(spec.get("plotLength",80))

    if sh in ("trapezoidal","trapezoid"):
        fw = float(spec.get("frontWidth", pw))
        bw = float(spec.get("backWidth",  pw))
        pw = max(fw, bw)
        poly = [(0,0),(fw,0),(bw,pl),(0,pl)]
     
        uw = min(fw, bw) - 2*M
        uh = pl - 2*M
        extras = {"frontWidth":fw,"backWidth":bw}

    elif sh in ("lshaped","lshape","lplot"):
        nw = float(spec.get("notchWidth",  pw * 0.35))
        nh = float(spec.get("notchLength", pl * 0.38))
        cor = spec.get("notchCorner","top-right").lower().replace(" ","")
        if   "right" in cor and "top" in cor:  poly=[(0,0),(pw-nw,0),(pw-nw,nh),(pw,nh),(pw,pl),(0,pl)]
        elif "left"  in cor and "top" in cor:  poly=[(nw,0),(pw,0),(pw,pl),(0,pl),(0,nh),(nw,nh)]
        elif "right" in cor and "bot" in cor:  poly=[(0,0),(pw,0),(pw,pl-nh),(pw-nw,pl-nh),(pw-nw,pl),(0,pl)]
        else:                                   poly=[(0,0),(pw,0),(pw,pl),(nw,pl),(nw,pl-nh),(0,pl-nh)]
        uw, uh = pw - 2*M, pl - 2*M
        extras = {"notchWidth":nw,"notchLength":nh,"notchCorner":cor}

    elif sh in ("triangular","triangle","pie","fan"):
        base = float(spec.get("plotBase",   pw))
        ht   = float(spec.get("plotHeight", pl))
        pw, pl = base, ht
  
        poly = [(0,pl),(base,pl),(base*0.12,0)]
       
        uw = base * 0.68 - 2*M
        uh = pl   * 0.72 - 2*M
        extras = {"plotBase":base,"plotHeight":ht}

    elif sh in ("cornerplot","corner","cornercut","truncated"):
        cut = float(spec.get("cornerCut", min(pw,pl)*0.14))
        poly = [(cut,0),(pw,0),(pw,pl),(0,pl),(0,cut)]
        uw, uh = pw - 2*M, pl - 2*M
        extras = {"cornerCut":cut}

    else:  
        sh = "rectangular"
        poly = [(0,0),(pw,0),(pw,pl),(0,pl)]
        uw, uh = pw - 2*M, pl - 2*M
        extras = {}

    return {
        "pw":pw, "pl":pl,
        "uw":max(uw, 10.0), "uh":max(uh, 10.0),
        "ox":M,  "oy":M,
        "poly":poly, "shape":sh, "extras":extras
    }


def layout_rooms(spec, variant=0):
    g = plot_geo(spec)
    uw, uh, ox, oy, sh = g["uw"], g["uh"], g["ox"], g["oy"], g["shape"]

    rooms  = spec.get("rooms", [])
    def bt(t):  return [dict(r) for r in rooms if r.get("type") == t]
    def g1(t):  a = bt(t); return a[0] if a else None

    garage  = g1("garage");   entry   = g1("entry");    study   = g1("study")
    living  = g1("living");   kitchen = g1("kitchen");  dining  = g1("dining")
    stair   = g1("staircase");laundry = g1("laundry")
    beds    = sorted(bt("bedroom"),  key=lambda r: 0 if r.get("isMaster") else 1)
    baths   = bt("bathroom")
    ensuite = next((b for b in baths if b.get("isEnsuite")), None)
    sbaths  = [b for b in baths if not b.get("isEnsuite")]
    master  = next((b for b in beds  if b.get("isMaster")), None)
    obeds   = [b for b in beds  if not b.get("isMaster")]
    nbed    = len(beds)
    aspect  = uw / max(uh, 1)

    placed = []

    HABITABLE = {"bedroom","living","kitchen","dining","study","entry"}

    def _min_w(r): return 7.0 if r.get("type") in HABITABLE else 3.5
    def _min_h(r): return 7.0 if r.get("type") in HABITABLE else 3.5

    def ph(items, sx, sy, sw, sh2):
        """Horizontal strip: widths proportional, heights = sh2. Zero gaps.
        IRC: habitable rooms get minimum 7ft width even after proportional scaling."""
        its = [r for r in items if r]
        if not its: return
        nat = [max(float(r.get("width",10)), _min_w(r)) for r in its]
        tot = sum(nat) or 1
        ws  = [sw * n / tot for n in nat]
      
        mins = [_min_w(r) for r in its]
        clamped = [max(w, m) for w, m in zip(ws, mins)]
        overflow = sum(clamped) - sw
        if overflow > 0.01:
            
            free_idx = [i for i,w in enumerate(ws) if w > mins[i] + 0.01]
            free_total = sum(ws[i] for i in free_idx) or 1
            for i in free_idx:
                clamped[i] = max(mins[i], ws[i] - overflow * ws[i]/free_total)
        clamped[-1] = sw - sum(clamped[:-1])   # fix rounding drift
        cx = sx
        for r, rw in zip(its, clamped):
            placed.append({**r,
                "x":round(cx,2),"y":round(sy,2),
                "w":round(rw,2),"h":round(sh2,2)})
            cx += rw

    def pv(items, sx, sy, sw, sh2):
        """Vertical column: heights proportional, widths = sw. Zero gaps."""
        its = [r for r in items if r]
        if not its: return
        nat = [max(float(r.get("length",10)), _min_h(r)) for r in its]
        tot = sum(nat) or 1
        hs  = [sh2 * n / tot for n in nat]
        hs[-1] = sh2 - sum(hs[:-1])
        cy = sy
        for r, rh in zip(its, hs):
            placed.append({**r,
                "x":round(sx,2),"y":round(cy,2),
                "w":round(sw,2),"h":round(rh,2)})
            cy += rh

    def p1(r, sx, sy, sw, sh2):
        if r:
            placed.append({**r,
                "x":round(sx,2),"y":round(sy,2),
                "w":round(sw,2),"h":round(sh2,2)})

    def corr(sx, sy, sw, sh2):
        """Corridor/hallway — IRC min 3ft wide."""
        sh2 = max(sh2, 3.0)
        placed.append({"type":"hallway","name":"Hallway",
            "x":round(sx,2),"y":round(sy,2),
            "w":round(sw,2),"h":round(sh2,2),
            "width":int(sw),"length":int(sh2)})

    def priv_strip():
        """Returns ordered list: master, ensuite, bed2, [bed3..], shared_bath, [laundry]"""
        z = []
        if master:  z.append(master)
        if ensuite: z.append(ensuite)   
        if len(obeds) >= 2:
            z.append(obeds[0])
            if sbaths: z.append(sbaths[0])
            z += obeds[1:]
            z += sbaths[1:]
        else:
            z += obeds
            z += sbaths
        if laundry: z.append(laundry)
        return z

    def zone_nat_h(room_list):
        """Average natural length of rooms in a zone."""
        its = [r for r in room_list if r]
        if not its: return 10.0
        return sum(r.get("length",10) for r in its) / len(its)

    def zone_nat_w(room_list):
        its = [r for r in room_list if r]
        if not its: return 10.0
        return sum(r.get("width",10) for r in its) / len(its)

    def compute_zone_heights(za_rooms, zb_rooms, zc_rooms, corridor_h=4.0):
        """
        Distribute uh proportionally to natural room depths.
        Guarantees za+zb+zc+corr = uh exactly.
        """
        ha = zone_nat_h(za_rooms)
        hb = zone_nat_h(zb_rooms)
        hc = zone_nat_h(zc_rooms)
        tot = ha + hb + hc or uh
        rem = uh - corridor_h
        za = max(8.0, round(rem * ha / tot, 2))
        zb = max(8.0, round(rem * hb / tot, 2))
        zc = rem - za - zb
        if zc < 8.0:      
            delta = 8.0 - zc
            za = max(6.0, za - delta * 0.5)
            zb = max(6.0, zb - delta * 0.5)
            zc = rem - za - zb
        return round(za,2), round(zb,2), round(corridor_h,2), round(zc,2)
    TYPO_NAMES = ["Linear","Single-Loaded","Central-Hall","Enfilade",
                  "L-Shape","Double-Loaded","Clustered-Core","Courtyard-Wrap"]

    def valid_typologies():
        valid = []
        for i, name in enumerate(TYPO_NAMES):
            if name == "L-Shape"       and sh not in ("rectangular","lshaped"): continue
            if name == "Courtyard-Wrap"and nbed < 2: continue
            if name == "Enfilade"      and nbed > 4: continue  
            if name == "Clustered-Core"and nbed < 3: continue
            valid.append(i)
        return valid

    if variant == 0:
        if sh == "lshaped":               t = 4
        elif sh == "triangular":          t = 1
        elif sh == "corner":              t = 7
        elif aspect >= 1.4 and nbed <= 3: t = 7
        elif aspect <= 0.70:              t = 1
        elif nbed >= 4:                   t = 6
        else:                             t = 0
    else:
        valid = valid_typologies()
        t = valid[(variant - 1) % len(valid)]

    spec["_typology"] = TYPO_NAMES[t]

    if t == 0:
        za = [garage, entry, study]
        zb = [living, stair, kitchen, dining]  
        zc = priv_strip()
        z1h, z2h, zhh, z3h = compute_zone_heights(za, zb, zc)
        ph(za,          ox, oy,                  uw, z1h)
        ph(zb,          ox, oy+z1h,              uw, z2h)
        corr(           ox, oy+z1h+z2h,          uw, zhh)
        ph(zc,          ox, oy+z1h+z2h+zhh,      uw, z3h)

   
    elif t == 1:
        lcW = round(uw * 0.26, 2);  rcW = uw - lcW;  rcX = ox + lcW
        gH  = round(uh * 0.44, 2);  sH  = uh - gH
        p1(garage, ox, oy,    lcW, gH)
        p1(study,  ox, oy+gH, lcW, sH)
        za = [entry]
        zb = [living, kitchen, dining, stair]
        zc = priv_strip()
        r1h, r2h, rhh, r3h = compute_zone_heights(za, zb, zc, corridor_h=3.5)
        ph(za, rcX, oy,              rcW, r1h)
        ph(zb, rcX, oy+r1h,          rcW, r2h)
        corr(  rcX, oy+r1h+r2h,      rcW, rhh)
        ph(zc, rcX, oy+r1h+r2h+rhh,  rcW, r3h)

    elif t == 2:
        za = [garage, entry, study]
        zb = [living, kitchen, dining]
        zc = priv_strip()
        z1h, z2h, zhh, z3h = compute_zone_heights(za, zb, zc)
        lw = round(uw*.30,2); mw = round(uw*.34,2); rw = uw-lw-mw
        p1(garage, ox,      oy,    lw, z1h)
        ph([entry,stair],   ox+lw, oy, mw, z1h)
        p1(study,  ox+lw+mw,oy,    rw, z1h)
        p1(living, ox,      oy+z1h,lw, z2h)
        placed.append({"type":"hallway","name":"Central Hall",
            "x":round(ox+lw,2),"y":round(oy+z1h,2),
            "w":round(mw,2),"h":round(z2h,2),"width":int(mw),"length":int(z2h)})
        ph([kitchen,dining],ox+lw+mw,oy+z1h,rw, z2h)
        corr(ox, oy+z1h+z2h, uw, zhh)
        mW = round(uw*.42,2)
        pv([master,ensuite],      ox,    oy+z1h+z2h+zhh, mW,    z3h)
        ph(obeds+sbaths+([laundry] if laundry else []),
           ox+mW, oy+z1h+z2h+zhh, uw-mW, z3h)
    elif t == 3:
        c1 = round(uw*.38,2); c2 = round(uw*.36,2); c3 = uw-c1-c2
        gH = round(uh*.22,2); lvH = round(uh*.40,2); msH = uh-gH-lvH
        p1(garage, ox,      oy,        c1, gH)
        p1(living, ox,      oy+gH,     c1, lvH)
        pv([master,ensuite],ox,         oy+gH+lvH, c1, msH)
        eH  = round(uh*.22,2); kdH = round(uh*.40,2); bdH = uh-eH-kdH
        p1(entry,           ox+c1, oy,     c2, eH)
        pv([kitchen,dining],ox+c1, oy+eH,  c2, kdH) 
        pv(obeds[:2],       ox+c1, oy+eH+kdH, c2, bdH)
        stH = round(uh*.22,2); scH = round(uh*.40,2)
        p1(study, ox+c1+c2, oy,        c3, stH)
        p1(stair, ox+c1+c2, oy+stH,    c3, scH)
        pv(sbaths+([laundry] if laundry else []),
           ox+c1+c2, oy+stH+scH, c3, uh-stH-scH)

    elif t == 4:
        hwH = round(uh*.38,2)
        priv = priv_strip()
        n_priv = max(1, len(priv))
        min_priv_w = n_priv * 7.0
        vwW = round(max(uw * 0.55, min(min_priv_w + 2, uw * 0.72)), 2)
        vwX = ox + uw - vwW;  vwH = uh - hwH;  svW = uw - vwW
        ph([garage,entry,living,stair,kitchen,dining], ox, oy, uw, hwH)
        zhL = max(round(vwH*.08,2), 3.0)
        corr(      vwX, oy+hwH,      vwW, zhL)
        ph(priv,   vwX, oy+hwH+zhL,  vwW, vwH-zhL)
        pv([r for r in [study] if r], ox, oy+hwH, svW, vwH)

    elif t == 5:
        spY = round(uh*.46,2); spH = max(round(uh*.07,2),3.0)
        topH = spY;  botH = uh-spY-spH
        endW = max(round(uw*.12,2), 7.0); midW = uw-2*endW  
        p1(entry, ox,          oy+spY, endW, spH)
        p1(stair, ox+uw-endW,  oy+spY, endW, spH)
        corr(ox+endW, oy+spY, midW, spH)
        ph([garage,living,kitchen,dining], ox, oy,        uw, topH)
        ph(priv_strip(),                  ox, oy+spY+spH, uw, botH)

    elif t == 6:
        c1 = round(uw*.36,2); c2 = round(uw*.26,2); c3 = uw-c1-c2
        r1 = round(uh*.18,2); r2 = round(uh*.22,2); r3 = round(uh*.22,2)
        r4 = uh-r1-r2-r3
       
        p1(garage, ox,       oy,      c1, r1)
        p1(entry,  ox+c1,    oy,      c2, r1)
        p1(study,  ox+c1+c2, oy,      c3, r1)
      
        p1(living,  ox,       oy+r1,   c1, r2)
        p1(laundry, ox+c1,    oy+r1,   c2, r2)
        p1(kitchen, ox+c1+c2, oy+r1,   c3, r2)
      
        p1(dining,  ox,       oy+r1+r2,c1, r3)
        p1(stair,   ox+c1,    oy+r1+r2,c2, r3)
        sb0 = sbaths[0] if sbaths else None
        p1(sb0,     ox+c1+c2, oy+r1+r2,c3, r3)
        # Row 4: private
        mW = round(uw*.40,2)
        corr(ox+mW, oy+r1+r2+r3, c2, r4)
        pv([master,ensuite],              ox,    oy+r1+r2+r3, mW,    r4)
        ph(obeds+sbaths[1:],              ox+mW+c2, oy+r1+r2+r3, uw-mW-c2, r4)

    
    elif t == 7:
        wW   = round(uw*.24,2); midW = uw-2*wW; topH = round(uh*.28,2)

        ph([entry,living,stair,kitchen,dining], ox, oy, uw, topH)
       
        botH = uh-topH
        gH   = round(botH*.50,2)
        p1(garage, ox, oy+topH,    wW, gH)
        p1(study,  ox, oy+topH+gH, wW, botH-gH)
        
        zhC = max(round(botH*.09,2),3.0)
        corr(ox+uw-wW, oy+topH,     wW, zhC)
        pv(priv_strip(), ox+uw-wW, oy+topH+zhC, wW, botH-zhC)
        placed.append({"type":"courtyard","name":"Courtyard / Garden",
            "x":round(ox+wW,2),"y":round(oy+topH,2),
            "w":round(midW,2),"h":round(botH,2),
            "width":int(midW),"length":int(botH)})

    for r in placed:
        if r.get("type","") in HABITABLE:
            r["_irc_warn"] = (r["w"] * r["h"]) < 69.0

    return placed

def render_svg(spec, placed):
    S     = 8      
    PAD   = 75    
    TH    = 60     

    g  = plot_geo(spec)
    pw, pl = g["pw"], g["pl"]
    tw = int(pw*S + PAD*2)
    th = int(pl*S + PAD*2 + TH)

    def px(x): return PAD + x*S
    def py(y): return PAD + y*S

    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{tw}" height="{th}" '
             f'style="background:#060e22;border-radius:8px;display:block;'
             f'font-family:\'Courier New\',monospace;">')

    for gx in range(0, int(pw)+1, 10):
        o.append(f'<line x1="{px(gx):.0f}" y1="{py(0)}" x2="{px(gx):.0f}" y2="{py(pl)}" '
                 f'stroke="rgba(40,100,210,.16)" stroke-width=".8"/>')
    for gy in range(0, int(pl)+1, 10):
        o.append(f'<line x1="{px(0)}" y1="{py(gy):.0f}" x2="{px(pw)}" y2="{py(gy):.0f}" '
                 f'stroke="rgba(40,100,210,.16)" stroke-width=".8"/>')
    for gx in range(0, int(pw)+1, 5):
        o.append(f'<line x1="{px(gx):.0f}" y1="{py(0)}" x2="{px(gx):.0f}" y2="{py(pl)}" '
                 f'stroke="rgba(40,100,210,.06)" stroke-width=".3"/>')
    for gy in range(0, int(pl)+1, 5):
        o.append(f'<line x1="{px(0)}" y1="{py(gy):.0f}" x2="{px(pw)}" y2="{py(gy):.0f}" '
                 f'stroke="rgba(40,100,210,.06)" stroke-width=".3"/>')

    poly_px = " ".join(f"{px(x):.1f},{py(y):.1f}" for x,y in g["poly"])
    o.append(f'<defs><clipPath id="pc">'
             f'<polygon points="{poly_px}"/></clipPath></defs>')
    o.append(f'<polygon points="{poly_px}" fill="rgba(8,18,58,.85)" stroke="none"/>')

    if g["shape"] == "triangular":
        base = g["extras"]["plotBase"]
        tip_pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x,y in
                  [(base*.12,0),(base*.55,pl*.35),(0,pl*.35),(0,0)])
        o.append(f'<polygon points="{tip_pts}" fill="rgba(18,65,18,.45)" '
                 f'stroke="#135516" stroke-width="1" stroke-dasharray="4,3"/>')
        o.append(f'<text x="{px(base*.12):.0f}" y="{py(pl*.16):.0f}" '
                 f'fill="rgba(55,155,55,.55)" font-size="9" text-anchor="middle">garden</text>')
    elif g["shape"] == "corner":
        cut = g["extras"]["cornerCut"]
        o.append(f'<polygon points="{px(0):.1f},{py(0):.1f} {px(cut):.1f},{py(0):.1f} '
                 f'{px(0):.1f},{py(cut):.1f}" fill="rgba(18,65,18,.4)" '
                 f'stroke="#135516" stroke-width="1" stroke-dasharray="3,3"/>')

    for r in placed:
        rx, ry  = px(r["x"]), py(r["y"])
        rw, rh  = r["w"]*S,   r["h"]*S
        rt      = r.get("type","default")
        c       = RC.get(rt, RC["default"])
        clip    = 'clip-path="url(#pc)"'

        o.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
                 f'fill="{c["f"]}" stroke="none" {clip}/>')

        if rt == "courtyard":
            cx2, cy2 = rx+rw/2, ry+rh/2
            o.append(f'<text x="{cx2:.0f}" y="{cy2-6:.0f}" text-anchor="middle" '
                     f'dominant-baseline="middle" fill="rgba(55,175,55,.6)" '
                     f'font-size="11" font-style="italic">Courtyard</text>')
            o.append(f'<text x="{cx2:.0f}" y="{cy2+10:.0f}" text-anchor="middle" '
                     f'dominant-baseline="middle" fill="rgba(55,175,55,.4)" '
                     f'font-size="8">/ Garden</text>')
            
            for hx in range(int(rx)+8, int(rx+rw)-8, 14):
                o.append(f'<line x1="{hx}" y1="{ry+4:.0f}" x2="{hx}" '
                         f'y2="{ry+rh-4:.0f}" stroke="rgba(40,120,40,.18)" stroke-width="1"/>')
            continue

        WT = 3  # wall thickness px
        for wx,wy,ww,wh in [(rx,ry,rw,WT),(rx,ry+rh-WT,rw,WT),
                             (rx,ry,WT,rh),(rx+rw-WT,ry,WT,rh)]:
            o.append(f'<rect x="{wx:.1f}" y="{wy:.1f}" width="{ww:.1f}" height="{wh:.1f}" '
                     f'fill="{c["w"]}" opacity=".95" {clip}/>')

        if rt == "hallway":
            my = ry + rh/2
            for ax in [rx+rw*.25, rx+rw*.5, rx+rw*.75]:
                o.append(f'<text x="{ax:.0f}" y="{my+4:.0f}" text-anchor="middle" '
                         f'fill="rgba(210,210,140,.28)" font-size="12">›</text>')
            o.append(f'<text x="{rx+rw/2:.0f}" y="{my+5:.0f}" text-anchor="middle" '
                     f'fill="rgba(200,200,130,.22)" font-size="6.5" letter-spacing="2.5">'
                     f'CIRCULATION</text>')
            continue

        if rt == "staircase":
            n_treads = max(4, min(10, int(rh / 8)))
            tread_h  = rh / (n_treads + 1)
            for i in range(1, n_treads+1):
                ty = ry + i * tread_h
                o.append(f'<line x1="{rx+WT:.1f}" y1="{ty:.1f}" '
                         f'x2="{rx+rw-WT:.1f}" y2="{ty:.1f}" '
                         f'stroke="{c["a"]}" stroke-width="1.2" {clip}/>')
            # Directional arrow pointing up (towards upper floor)
            ax2 = rx + rw/2; ay1 = ry+rh-12; ay2 = ry+10
            o.append(f'<line x1="{ax2:.1f}" y1="{ay1:.1f}" x2="{ax2:.1f}" y2="{ay2:.1f}" '
                     f'stroke="{c["a"]}" stroke-width="1.5" {clip}/>')
            o.append(f'<polygon points="{ax2:.1f},{ay2:.1f} '
                     f'{ax2-5:.1f},{ay2+9:.1f} {ax2+5:.1f},{ay2+9:.1f}" '
                     f'fill="{c["a"]}" {clip}/>')

        elif rt == "bathroom":
            # Toilet: oval; Bathtub: rectangle; Sink: circle (standard symbols)
            # Scale fixtures to room size
            scale = min(rw, rh) / 60
            # Toilet (bottom-right)
            tx = rx + rw*0.65; ty = ry + rh*0.55
            tw2 = max(10, min(20, rw*0.25)); tl = max(14, min(26, rh*0.30))
            o.append(f'<ellipse cx="{tx:.1f}" cy="{ty+tl*.4:.1f}" rx="{tw2*.45:.1f}" '
                     f'ry="{tl*.5:.1f}" fill="none" stroke="{c["a"]}" '
                     f'stroke-width="1.2" {clip}/>')
            o.append(f'<rect x="{tx-tw2*.5:.1f}" y="{ty-4:.1f}" width="{tw2:.1f}" '
                     f'height="8" rx="2" fill="none" stroke="{c["a"]}" '
                     f'stroke-width="1.2" {clip}/>')
            # Bathtub (left side) — only if room is wide enough
            if rw > 44:
                bx = rx + WT + 3; by = ry + WT + 3
                bw2 = rw * 0.35; bh2 = rh * 0.5
                o.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw2:.1f}" '
                         f'height="{bh2:.1f}" rx="4" fill="none" '
                         f'stroke="{c["a"]}" stroke-width="1.2" {clip}/>')
                # Drain circle
                o.append(f'<circle cx="{bx+bw2*.5:.1f}" cy="{by+bh2*.8:.1f}" r="3" '
                         f'fill="none" stroke="{c["a"]}" stroke-width="1" {clip}/>')
            # Sink circle (top-right)
            if rh > 36:
                sx2 = rx + rw*0.7; sy2 = ry + WT + 8
                o.append(f'<circle cx="{sx2:.1f}" cy="{sy2:.1f}" r="7" fill="none" '
                         f'stroke="{c["a"]}" stroke-width="1.2" {clip}/>')

        elif rt == "kitchen":
            # Counter rectangle along top wall + island if hasIsland
            ctr_h = max(8, rh * 0.18)
            o.append(f'<rect x="{rx+WT:.1f}" y="{ry+WT:.1f}" '
                     f'width="{rw-WT*2:.1f}" height="{ctr_h:.1f}" '
                     f'fill="none" stroke="{c["a"]}" stroke-width="1.2" {clip}/>')
            # Sink in counter
            o.append(f'<rect x="{rx+rw*.6:.1f}" y="{ry+WT+2:.1f}" '
                     f'width="{rw*.2:.1f}" height="{ctr_h-4:.1f}" rx="2" '
                     f'fill="none" stroke="{c["a"]}" stroke-width="1" {clip}/>')
            # Island (if hasIsland)
            if r.get("hasIsland") and rh > 55:
                iw = rw * 0.55; ih = rh * 0.24
                ix = rx + (rw - iw)/2; iy = ry + rh*0.5
                o.append(f'<rect x="{ix:.1f}" y="{iy:.1f}" width="{iw:.1f}" '
                         f'height="{ih:.1f}" rx="2" fill="none" '
                         f'stroke="{c["a"]}" stroke-width="1.5" stroke-dasharray="4,2" {clip}/>')

        elif rt == "bedroom":
            bw2 = min(rw*0.65, 52); bh2 = min(rh*0.45, 40)
            bx2 = rx + (rw-bw2)/2; by2 = ry + rh*0.45
            o.append(f'<rect x="{bx2:.1f}" y="{by2:.1f}" width="{bw2:.1f}" '
                     f'height="{bh2:.1f}" rx="3" fill="none" '
                     f'stroke="{c["a"]}" stroke-width="1.3" {clip}/>')
            # Pillow(s)
            pw2 = bw2*.22
            if r.get("isMaster"):  # two pillows
                for px3 in [bx2+bw2*.15, bx2+bw2*.63]:
                    o.append(f'<rect x="{px3:.1f}" y="{by2+3:.1f}" '
                             f'width="{pw2:.1f}" height="{min(8, bh2*.28):.1f}" rx="2" '
                             f'fill="none" stroke="{c["a"]}" stroke-width="1" {clip}/>')
            else:  # one pillow
                o.append(f'<rect x="{bx2+bw2*.38:.1f}" y="{by2+3:.1f}" '
                         f'width="{pw2:.1f}" height="{min(8, bh2*.28):.1f}" rx="2" '
                         f'fill="none" stroke="{c["a"]}" stroke-width="1" {clip}/>')

        elif rt == "garage":
            # Car outline(s)
            cars = r.get("cars", 1)
            car_w = (rw - WT*2 - (cars-1)*4) / cars if cars > 0 else rw*0.8
            for ci in range(cars):
                cx3 = rx + WT + ci*(car_w+4)
                cy3 = ry + rh*0.25
                ch2 = rh*0.55
                o.append(f'<rect x="{cx3:.1f}" y="{cy3:.1f}" width="{car_w:.1f}" '
                         f'height="{ch2:.1f}" rx="4" fill="none" '
                         f'stroke="{c["a"]}" stroke-width="1.2" stroke-dasharray="3,2" {clip}/>')

        dw = min(max(rw * 0.32, 14), 30)
        dx = rx + WT; dy = ry + rh - WT
        o.append(f'<line x1="{dx:.1f}" y1="{dy:.1f}" '
                 f'x2="{dx+dw:.1f}" y2="{dy:.1f}" '
                 f'stroke="{c["s"]}" stroke-width="1.8" {clip}/>')
        o.append(f'<path d="M {dx:.1f} {dy:.1f} A {dw:.1f} {dw:.1f} 0 0 0 '
                 f'{dx:.1f} {dy-dw:.1f}" fill="none" stroke="{c["s"]}" '
                 f'stroke-width="1" stroke-dasharray="3,2" {clip}/>')

        M2 = g["ox"] + 0.8
        is_t = r["y"] <= M2;           is_b = (r["y"]+r["h"]) >= (pl - M2)
        is_l = r["x"] <= M2;           is_r = (r["x"]+r["w"]) >= (pw - M2)
        wm = rw * 0.18; ws2 = rx+wm; we2 = rx+rw-wm
        def win_h(wy, inside=False):
            # 3 lines: outer solid, middle gap, inner solid
            for off, op, sw2 in [(0,1.0,2.0),(3,0.5,1.0),(-3,0.5,1.0)]:
                o.append(f'<line x1="{ws2:.1f}" y1="{wy+off:.1f}" '
                         f'x2="{we2:.1f}" y2="{wy+off:.1f}" '
                         f'stroke="rgba(170,225,255,{op})" stroke-width="{sw2}" {clip}/>')
        def win_v(wx):
            wy1, wy2 = ry+rh*.22, ry+rh*.78
            for off, op, sw2 in [(0,1.0,2.0),(3,0.5,1.0),(-3,0.5,1.0)]:
                o.append(f'<line x1="{wx+off:.1f}" y1="{wy1:.1f}" '
                         f'x2="{wx+off:.1f}" y2="{wy2:.1f}" '
                         f'stroke="rgba(170,225,255,{op})" stroke-width="{sw2}" {clip}/>')
        if is_t: win_h(ry + 1)
        if is_b: win_h(ry + rh - 1)
        if is_l: win_v(rx + 1)
        if is_r: win_v(rx + rw - 1)

        # ── FIX C7: Room label — 3-tier font scaling, no ugly abbreviations ─
        cx2, cy2 = rx+rw/2, ry+rh/2
        name = r.get("name", rt.title())
        wf   = r.get("w", r.get("width", 0))
        hf   = r.get("h", r.get("length", 0))
        area = round(wf * hf)

        # Tier 1: normal (room > 70×40px), Tier 2: compact (>40×28), Tier 3: tiny
        if rw >= 70 and rh >= 40:
            fsN = min(13, max(8, int(rw/7.5))); fsD = fsN-2; fsA = fsN-4
            # Split long names at space
            words = name.split()
            if len(words) > 2 and rw < 90:
                mid = len(words)//2
                line1 = " ".join(words[:mid]); line2 = " ".join(words[mid:])
                o.append(f'<text x="{cx2:.0f}" y="{cy2-12:.0f}" text-anchor="middle" '
                         f'dominant-baseline="middle" fill="rgba(255,255,255,.94)" '
                         f'font-size="{fsN}" font-weight="bold">{line1}</text>')
                o.append(f'<text x="{cx2:.0f}" y="{cy2-1:.0f}" text-anchor="middle" '
                         f'dominant-baseline="middle" fill="rgba(255,255,255,.94)" '
                         f'font-size="{fsN}" font-weight="bold">{line2}</text>')
            else:
                o.append(f'<text x="{cx2:.0f}" y="{cy2-8:.0f}" text-anchor="middle" '
                         f'dominant-baseline="middle" fill="rgba(255,255,255,.94)" '
                         f'font-size="{fsN}" font-weight="bold">{name}</text>')
            o.append(f'<text x="{cx2:.0f}" y="{cy2+8:.0f}" text-anchor="middle" '
                     f'dominant-baseline="middle" fill="rgba(175,210,255,.75)" '
                     f'font-size="{fsD}">{wf:.0f}\'×{hf:.0f}\'</text>')
            if rh >= 50:
                o.append(f'<text x="{cx2:.0f}" y="{cy2+20:.0f}" text-anchor="middle" '
                         f'dominant-baseline="middle" fill="rgba(130,185,255,.48)" '
                         f'font-size="{fsA}">{area} sf</text>')

        elif rw >= 40 and rh >= 28:
            fsN = max(7, int(rw/9))
            o.append(f'<text x="{cx2:.0f}" y="{cy2-5:.0f}" text-anchor="middle" '
                     f'dominant-baseline="middle" fill="rgba(255,255,255,.90)" '
                     f'font-size="{fsN}" font-weight="bold">{name}</text>')
            o.append(f'<text x="{cx2:.0f}" y="{cy2+6:.0f}" text-anchor="middle" '
                     f'dominant-baseline="middle" fill="rgba(175,210,255,.68)" '
                     f'font-size="{max(6,fsN-2)}">{wf:.0f}\'×{hf:.0f}\'</text>')

        else:  # tiny rooms — just name, small font
            o.append(f'<text x="{cx2:.0f}" y="{cy2+2:.0f}" text-anchor="middle" '
                     f'dominant-baseline="middle" fill="rgba(255,255,255,.85)" '
                     f'font-size="7" font-weight="bold">{name}</text>')

    o.append(f'<polygon points="{poly_px}" fill="none" stroke="#4a9eff" stroke-width="4.8"/>')
    # Surveyor dashed line offset
    o.append(f'<polygon points="{poly_px}" fill="none" stroke="rgba(74,158,255,.22)" '
             f'stroke-width="1.5" stroke-dasharray="9,5" transform="translate(-3,-3)"/>')

    dc = "#4a9eff"; tk = 7
    # Width (top)
    o.append(f'<line x1="{px(0)}" y1="{py(0)-30}" x2="{px(pw)}" y2="{py(0)-30}" '
             f'stroke="{dc}" stroke-width="1"/>')
    for ex in [px(0), px(pw)]:
        o.append(f'<line x1="{ex}" y1="{py(0)-30-tk}" x2="{ex}" y2="{py(0)-30+tk}" '
                 f'stroke="{dc}" stroke-width="1.3"/>')
    o.append(f'<text x="{(px(0)+px(pw))/2:.0f}" y="{py(0)-40}" text-anchor="middle" '
             f'fill="{dc}" font-size="11">{pw:.0f}\'-0"</text>')
    # Depth (left)
    o.append(f'<line x1="{px(0)-30}" y1="{py(0)}" x2="{px(0)-30}" y2="{py(pl)}" '
             f'stroke="{dc}" stroke-width="1"/>')
    for ey in [py(0), py(pl)]:
        o.append(f'<line x1="{px(0)-30-tk}" y1="{ey}" x2="{px(0)-30+tk}" y2="{ey}" '
                 f'stroke="{dc}" stroke-width="1.3"/>')
    mh = (py(0)+py(pl))/2
    o.append(f'<text x="{px(0)-44}" y="{mh:.0f}" text-anchor="middle" fill="{dc}" '
             f'font-size="11" transform="rotate(-90,{px(0)-44},{mh:.0f})">'
             f'{pl:.0f}\'-0"</text>')

    ZM = {"garage":"FRONT/SERVICE","entry":"FRONT/SERVICE","study":"FRONT/SERVICE",
          "living":"PUBLIC LIVING","kitchen":"PUBLIC LIVING","dining":"PUBLIC LIVING",
          "staircase":"PUBLIC LIVING","hallway":"CIRCULATION",
          "bedroom":"PRIVATE","bathroom":"PRIVATE","laundry":"PRIVATE",
          "courtyard":"OUTDOOR"}
    ZC = {"FRONT/SERVICE":"rgba(148,155,182,.65)","PUBLIC LIVING":"rgba(255,88,88,.65)",
          "CIRCULATION":"rgba(220,220,140,.55)","PRIVATE":"rgba(50,138,255,.65)",
          "OUTDOOR":"rgba(50,170,50,.65)"}
    seen_z = {}
    for r in placed:
        z = ZM.get(r.get("type",""),"")
        if z and z not in seen_z: seen_z[z] = py(r["y"]) + 10
    for lbl, yp in seen_z.items():
        o.append(f'<text x="6" y="{yp:.0f}" fill="{ZC.get(lbl,"#aaa")}" '
                 f'font-size="7" writing-mode="tb" letter-spacing="1.8">{lbl}</text>')

    nax, nay = tw-40, PAD+40
    o.append(f'<circle cx="{nax}" cy="{nay}" r="18" fill="none" '
             f'stroke="rgba(74,158,255,.38)" stroke-width="1"/>')
    o.append(f'<polygon points="{nax},{nay-22} {nax-8},{nay+13} '
             f'{nax},{nay+6} {nax+8},{nay+13}" fill="#4a9eff"/>')
    o.append(f'<polygon points="{nax},{nay+6} {nax+8},{nay+13} '
             f'{nax},{nay-22}" fill="rgba(0,15,55,.5)"/>')
    o.append(f'<text x="{nax}" y="{nay-27}" text-anchor="middle" '
             f'fill="#4a9eff" font-size="13" font-weight="bold">N</text>')

    tby  = PAD + int(pl*S) + 12
    tbw  = int(pw*S)
    style_lbl = spec.get("style","custom").upper()
    typo  = spec.get("_typology","Linear")
    sh_lbl = g["shape"].replace("lshaped","L-Shaped").replace("cornerplot","Corner").title()
    today = date.today().strftime("%b %d, %Y")
    scl   = f'1" = {12/S:.1f}\''

    o.append(f'<rect x="{PAD}" y="{tby}" width="{tbw}" height="{TH-8}" '
             f'fill="rgba(10,24,75,.82)" stroke="#4a9eff" stroke-width="1"/>')
    # Dividers
    for fr in [0.44, 0.68, 0.84]:
        dx = PAD + int(tbw*fr)
        o.append(f'<line x1="{dx}" y1="{tby}" x2="{dx}" y2="{tby+TH-8}" '
                 f'stroke="#4a9eff" stroke-width=".8"/>')

    o.append(f'<text x="{PAD+9}" y="{tby+16}" fill="white" '
             f'font-size="12" font-weight="bold">'
             f'DREAMHOUSE AI — {style_lbl} RESIDENCE</text>')
    o.append(f'<text x="{PAD+9}" y="{tby+30}" '
             f'fill="rgba(155,205,255,.82)" font-size="9">'
             f'PLOT: {pw:.0f}\'×{pl:.0f}\' ({sh_lbl}) | AREA: {spec.get("houseArea",0)} SF | '
             f'{spec.get("floors",1)} FLOOR(S) | TYPOLOGY: {typo}</text>')
    o.append(f'<text x="{PAD+9}" y="{tby+44}" '
             f'fill="rgba(100,160,255,.48)" font-size="7.5">'
             f'Sources: Ching · Neufert · Schneider · Saratsis 2015 · IRC 2021 · White 1986</text>')

    m1 = PAD + int(tbw*0.56)
    o.append(f'<text x="{m1}" y="{tby+18}" text-anchor="middle" '
             f'fill="rgba(155,205,255,.82)" font-size="9">DATE: {today}</text>')
    o.append(f'<text x="{m1}" y="{tby+32}" text-anchor="middle" '
             f'fill="rgba(155,205,255,.82)" font-size="9">SCALE: {scl}</text>')

    m2 = PAD + int(tbw*0.76)
    o.append(f'<text x="{m2}" y="{tby+18}" text-anchor="middle" '
             f'fill="rgba(155,205,255,.82)" font-size="9">FLOOR PLAN · LEVEL 1</text>')

    m3 = PAD + int(tbw*0.92)
    o.append(f'<text x="{m3}" y="{tby+18}" text-anchor="middle" '
             f'fill="rgba(155,205,255,.82)" font-size="9">DREAMHOUSE AI</text>')

    o.append("</svg>")
    return "\n".join(o)

def parse_bp(text):
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if not m: return None
    try:    return json.loads(m.group(1))
    except: return None

def call_ai(history):
    try:
        k = os.environ.get("GROQ_API_KEY","")
        if not k:
            return ("⚠️ **GROQ_API_KEY not set.**\n\n"
                    "Run `export GROQ_API_KEY=your_key` then restart Streamlit.\n"
                    "Get a free key (14,400 req/day) at https://console.groq.com")
        client = Groq(api_key=k)
        msgs = [{"role":"system","content":SYSTEM_PROMPT}] + \
               [{"role":m["role"],"content":m["content"]} for m in history]
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=msgs, max_tokens=1400)
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ API Error: {e}"

if "dh_messages" not in st.session_state:
    st.session_state.dh_messages = [{"role":"assistant","content":
        "Welcome to **Dreamhouse AI** 🏡\n\n"
        "I'll design your personalised floor plan using principles from "
        "*Ching, Neufert, and Schneider* — with 8 architectural typologies, "
        "5 plot shapes, and correct room adjacencies.\n\n"
        "**Let's start with your plot:**\n"
        "1. What shape is your plot? *(rectangular / L-shaped / trapezoidal / triangular / corner)*\n"
        "2. What are the dimensions? *(e.g. 60×80 feet)*\n"
        "3. How many floors?"}]
    st.session_state.dh_spec    = None
    st.session_state.dh_placed  = None
    st.session_state.dh_variant = 0

st.markdown(
    '<span style="font-family:Courier New,monospace;font-size:1.3rem;'
    'font-weight:700;color:#4a9eff;letter-spacing:.06em;">⬡ DREAMHOUSE '
    '<span style="color:#fff">AI</span></span>'
    '&nbsp;<span style="font-size:.68rem;color:rgba(150,190,255,.38);'
    'letter-spacing:.16em;">ARCHITECTURAL BLUEPRINT GENERATOR</span>',
    unsafe_allow_html=True)
st.divider()

chat_col, bp_col = st.columns([1, 1.45])
with chat_col:
    chat_box = st.container(height=530)
    with chat_box:
        for msg in st.session_state.dh_messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="ub">{msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                disp = msg["content"]
                if "BLUEPRINT_READY" in disp:
                    disp = re.sub(r"BLUEPRINT_READY[\s\S]*",
                                  "✅ **Blueprint generated!** Floor plan is on the right →", disp)
                st.markdown(f'<div class="ab">{disp}</div>',
                            unsafe_allow_html=True)

    user_in = st.chat_input("Describe your dream home...")
    if user_in:
        st.session_state.dh_messages.append({"role":"user","content":user_in})
        with st.spinner("Thinking..."):
            reply = call_ai(list(st.session_state.dh_messages))
        st.session_state.dh_messages.append({"role":"assistant","content":reply})
        if "BLUEPRINT_READY" in reply:
            spec = parse_bp(reply)
            if spec:
                st.session_state.dh_spec    = spec
                st.session_state.dh_placed  = layout_rooms(spec, 0)
                st.session_state.dh_variant = 0
        st.rerun()

with bp_col:
    if st.session_state.dh_spec and st.session_state.dh_placed:
        spec  = st.session_state.dh_spec
        typo  = spec.get("_typology","Layout")

        hdr, btn_col = st.columns([3,1])
        with hdr:
            st.markdown(f"**{spec.get('style','Custom').title()} Residence — {typo} Typology**")
        with btn_col:
            if st.button("🔄 Regenerate"):
                st.session_state.dh_variant += 1
                st.session_state.dh_placed = layout_rooms(
                    spec, st.session_state.dh_variant)
                st.rerun()

        leg = '<div style="display:flex;flex-wrap:wrap;gap:4px 10px;margin-bottom:5px;">'
        for t, c in RC.items():
            if t in ("default","hallway","courtyard"): continue
            leg += (f'<div style="display:flex;align-items:center;gap:3px;">'
                    f'<div style="width:11px;height:11px;border-radius:2px;'
                    f'background:{c["f"]};border:1.5px solid {c["s"]};"></div>'
                    f'<span style="font-size:.63rem;color:rgba(150,190,255,.58);'
                    f'font-family:Courier New,monospace;">{t.title()}</span></div>')
        leg += "</div>"
        st.markdown(leg, unsafe_allow_html=True)
        TDESC = {
            "Linear":        "Ching FSO Ch.5 · Linear org · public→private front-to-back",
            "Single-Loaded": "Schneider FPM p.18 · Side column · rooms face exterior",
            "Central-Hall":  "Schneider · Four-Square · centralised hall core",
            "Enfilade":      "Schneider p.22 · Baroque · room-to-room, no corridor",
            "L-Shape":       "Ching FSO p.141 · L-shaped wings · outdoor enclosure",
            "Double-Loaded": "Schneider / Saratsis 2015 · central corridor spine",
            "Clustered-Core":"Saratsis 2015 / Ebner 2010 · service core, rooms cluster",
            "Courtyard-Wrap":"Neufert Block / Ching FSO p.158 · U-shape courtyard",
        }
        st.markdown(
            f'<div style="font-size:.65rem;color:rgba(100,160,255,.42);'
            f'font-family:Courier New,monospace;margin-bottom:5px;">'
            f'📐 {TDESC.get(typo,"")}</div>', unsafe_allow_html=True)

        # Blueprint
        svg = render_svg(spec, st.session_state.dh_placed)
        st.markdown(f'<div style="overflow:auto;max-height:640px;">{svg}</div>',
                    unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="height:530px;display:flex;flex-direction:column;
             align-items:center;justify-content:center;
             color:rgba(74,158,255,.24);font-family:'Courier New',monospace;
             text-align:center;border:1px dashed rgba(74,158,255,.11);
             border-radius:10px;margin-top:1rem;">
          <div style="font-size:3rem;margin-bottom:16px;">📐</div>
          <div style="font-size:1.05rem;letter-spacing:.13em;margin-bottom:12px;">
            BLUEPRINT AWAITING</div>
          <div style="font-size:.68rem;color:rgba(150,190,255,.2);
               max-width:330px;line-height:2.1;">
            8 architectural typologies<br>
            5 plot shapes (rectangular · L-shape · trapezoid · triangle · corner)<br>
            Correct adjacencies · IRC 2021 compliance<br>
            Ching · Neufert · Schneider · Saratsis · White<br><br>
            ← Chat with your AI architect to begin
          </div>
        </div>""", unsafe_allow_html=True)
