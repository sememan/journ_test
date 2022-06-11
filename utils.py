import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px



def colorize(par, par_alpha):
    alphawert = par_alpha
    switcher={
        "DCS":"rgba(240,128,128,"+alphawert+")",#'lightcoral',
        "DCS_ENTRY":"rgba(240,128,128,"+alphawert+")",#'lightcoral',
        "DCS_PRELOGIN":"rgba(240,128,128,"+alphawert+")",#'lightcoral',
        "DCS_PERSONAL":"rgba(240,128,128,"+alphawert+")",#'lightcoral',
        "DCS_PAYMENT":"rgba(240,128,128,"+alphawert+")",#'lightcoral',
        "DCS_OVERVIEW":"rgba(240,128,128,"+alphawert+")",#'lightcoral',
        "DCS_CONFIRM":"rgba(240,16,16,"+alphawert+")",#'lightcoral',
        "konfigurator":"rgba(102,205,170,"+alphawert+")",#'mediumaquamarine',
        "CON":"rgba(102,205,170,"+alphawert+")",#'mediumaquamarine',
        "IUCP":"rgba(238,232,170,"+alphawert+")",#'magenta', #gold',
        "NCS":"rgba(252,169,133,"+alphawert+")",#'magenta', #gold',
        "SL_SIM":"rgba(238,232,170,"+alphawert+")",#'magenta', #gold',
        "SL_SEARCH":"rgba(238,232,170,"+alphawert+")",#'magenta', #gold',
        "SL_DEALER":"rgba(238,232,170,"+alphawert+")",#'magenta', #gold',
        "SL_VEH":"rgba(238,232,170,"+alphawert+")",#'magenta', #gold',
        "SL_HOME":"rgba(238,232,170,"+alphawert+")",#'magenta', #gold',
        "SHOWROOM":"rgba(100,149,237,"+alphawert+")",#'cornflowerblue',
        "NSC":"rgba(154,206,223,"+alphawert+")",#'cornflowerblue',
        "TOPICS":"rgba(165,137,193,"+alphawert+")",#'cornflowerblue',
        "HOME":"rgba(192,186,153,"+alphawert+")",#'lightgoldenrodyellow',
        "FORM":"rgba(240,128,128,"+alphawert+")",#floralwhite
        "DLO":"rgba(131,105,83,"+alphawert+")",#floralwhite
        #"S":"rgba(240,64,64,"+alphawert+")",#floralwhite
        "F_RFO":"rgba(240,16,16,"+alphawert+")",#floralwhite
        "S_RFO":"rgba(240,128,128,"+alphawert+")",#floralwhite
        "F_TDA":"rgba(240,16,16,"+alphawert+")",#floralwhite
        "S_TDA":"rgba(240,128,128,"+alphawert+")",#floralwhite
        "F_RFI":"rgba(240,16,16,"+alphawert+")",#floralwhite
        "S_RFI":"rgba(240,128,128,"+alphawert+")",#floralwhite
        "ENTRY":"rgba(255,250,240,"+alphawert+")",#floralwhite
        "EXIT":"rgba(255,250,240,"+alphawert+")"#floralwhite
    }
    return switcher.get(par, "rgba(192,192,192,"+alphawert+")") #silver


#############################


def df_filters(df, start_date='2021-09-01', end_date='2021-09-07',
               section_ref='FORM', form_started='None', form_finished='True'):
    print(f"params: dates{(start_date, end_date)}, section_ref='{section_ref}', " +
          f"form_started='{form_started}', form_finished='{form_finished}'")
    print('df', df.shape, df['MIDVN'].nunique(), end=' -> ')

    ### filters

    # dates
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    # section
    df = df.loc[[section_ref in j for j in df['Journey']], :].copy()

    # form
    if form_started != 'None':
        df = df[df['form_started'].astype(str) == form_started].copy()
    if form_finished != 'None':
        df = df[df['form_finished'].astype(str) == form_finished].copy()

    print(df.shape, df['MIDVN'].nunique())

    return df


def df_sankey(df, section_ref='FORM', NUM_MAXSTEPS=13, NUM_MINTRANS=3, ptype='linear'):
    NUM_MAXSTEPS = NUM_MAXSTEPS - 1

    start_time = time.time()
    print('df', df.shape, df['MIDVN'].nunique())

    dfv = df.groupby(['MIDVN', 'form_started', 'form_finished', 'date', 'Journey'])[
        'Journey_S_diff'].min().reset_index()
    print('dfv', dfv.shape, dfv['MIDVN'].nunique())

    ### stp_

    # pivot index="MIDVN", column=steps(01,02,03,...), value="Journey_S" (from fisrt to last section visit)
    stp_ = pd.DataFrame(df.groupby('MIDVN')['Journey_S'].apply(lambda x: list(x)).tolist())
    stp_ = stp_.rename(columns={c: 'stp_' + (str(c).zfill(2)) for c in stp_.columns})
    print('stp', stp_.shape)
    # select the first step occurrence of the reference section ("section_ref")
    stp_['stp_max'] = (stp_ == section_ref).idxmax(1)
    # replace with null values all steps after the reference section
    stp_ = stp_.apply(lambda x: x.loc[[c for c in stp_.columns if c <= x['stp_max']]], axis=1)  ### <- the slowest !!!
    if NUM_MAXSTEPS == 'max':
        NUM_MAXSTEPS = stp_.shape[1] - 1
    elif NUM_MAXSTEPS > stp_.shape[1]:
        NUM_MAXSTEPS = stp_.shape[1] - 1
        # reverse the step position (from ref to first section visit) - the step name remains unchanged!
    stp_ = pd.DataFrame(stp_.apply(lambda x: x[~x.isnull()].tolist()[::-1][:NUM_MAXSTEPS + 1], axis=1).tolist())
    stp_ = stp_.rename(columns={c: 'stp_' + (str(c).zfill(2)) for c in stp_.columns})
    print('stp', stp_.shape)
    stp_ = stp_.rename(columns={'stp_00': 'ref_point'})
    stp_ = stp_[stp_['ref_point'] == section_ref]
    print('stp', stp_.shape)
    assert len(set(stp_['ref_point'])) == 1
    # merge the steps into dfv and set index
    dfv = dfv.merge(stp_, left_index=True, right_index=True)
    dfv = dfv.set_index('MIDVN')
    assert stp_.shape[1] > 1
    print(stp_.head())

    print('dfv', dfv.shape, 'pre - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### cnt_

    # count the transitions between the steps
    for i in range(1, NUM_MAXSTEPS + 1):
        gbnames = ['ref_point', ]
        gbnames.extend(['stp_' + (str(i).zfill(2)) for i in range(1, i + 1)])
        dfv['cnt_' + (str(i).zfill(2))] = dfv.groupby(gbnames)['Journey_S_diff'].transform(sum)

    print('dfv', dfv.shape, 'cnt - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### jcode_

    # prepare the mapping dict
    tmp_keys = list(df['Journey_S'].unique())
    tmp_keys.append("ENTRY")
    tmp_keys.append("EXIT")
    tmp_values = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")[:len(tmp_keys)]
    map_dict = dict(zip(tmp_keys, tmp_values))

    # use the mapping dict to create a unique code for each step ("jcode_")
    for i in range(1, NUM_MAXSTEPS + 1):
        if i == 1:
            dfv["jcode_01"] = dfv.stp_01.map(map_dict)
        else:
            dfv["jcode_" + str(i).zfill(2)] = dfv["jcode_" + str(i - 1).zfill(2)] + (
                dfv["stp_" + str(i).zfill(2)].map(map_dict))

    print('dfv', dfv.shape, 'jcode - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### jrn_ = stp_ + jcode_ (unique id)

    # create a journey step code which identifies uniquely the step with the section
    dfv["jrn_00"] = dfv.ref_point
    for i in range(1, NUM_MAXSTEPS + 1):
        dfv["jrn_" + str(i).zfill(2)] = dfv["stp_" + str(i).zfill(2)] + "~" + dfv["jcode_" + str(i).zfill(2)]

    print('dfv', dfv.shape, 'jrn - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### transitions (with amount): source (src) > target (dest)

    # extrapolate the transitions between steps
    ll = []
    for i in range(0, NUM_MAXSTEPS):
        ll.append(list(np.where(dfv["jrn_" + str(i).zfill(2)].notnull() & dfv["jrn_" + str(i + 1).zfill(2)].notnull(), \
                                dfv["jrn_" + str(i).zfill(2)] + ' > ' + dfv["jrn_" + str(i + 1).zfill(2)] + ' > ' + \
                                dfv["cnt_" + str(i + 1).zfill(2)].astype(str), None)))

    fll = [item for sublist in ll for item in sublist]
    fll = list(set(fll))
    fll = list(filter(None, fll))

    tdf = pd.DataFrame(fll, columns=["transitions"])
    tdf["src"] = tdf["transitions"].str.split(pat=' > ').str[0]
    tdf["dest"] = tdf["transitions"].str.split(pat=' > ').str[1]
    tdf["amount"] = tdf["transitions"].str.split(pat=' > ').str[2]
    tdf["lsrc"] = tdf["src"].str.split(pat="~").str[0]
    tdf["ldest"] = tdf["dest"].str.split(pat="~").str[0]

    print('tdf', tdf.shape, 'transitions - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### remove amounts <3

    # remove transition steps which count less than 3 transitions
    tdf['amount'] = tdf['amount'].astype(float)
    tdf = tdf[tdf['amount'] > NUM_MINTRANS].copy()
    assert len(tdf) > 0

    print('tdf', tdf.shape, 'remove<3 - time:', round((time.time() - start_time) / 60, 2), 'min')

    if ptype == 'linear':

        ### concat ENTRY

        # introduce a dummy "ENTRY" section at the beginning of each journey
        dll = []
        for item in list(set(tdf.dest.values).difference(set(tdf.src.values))):
            dll.append([item + " > ENTRY > 1.0", item, "ENTRY", 1.0, item.split("~")[0], "ENTRY"])
        tdf = pd.concat([tdf, pd.DataFrame(dll, columns=tdf.columns)])

        print('tdf', tdf.shape, 'ENTRY - time:', round((time.time() - start_time) / 60, 2), 'min')

        ### final tdf

        # list of all section steps
        keys = sorted(set(tdf["src"].tolist() + tdf["dest"].tolist()))
        values = list(range(0, len(keys)))
        m = dict(zip(keys, values))

        # assign an unique integer (id) to the steps and order the tdf
        tdf["src_id"] = tdf["src"].map(m)
        tdf["dest_id"] = tdf["dest"].map(m)
        tdf["srcn"] = tdf["src"].str.split(pat="~").str[1]
        tdf['srcl'] = tdf['srcn'].str.len() / 10
        tdf.sort_values(by=["srcl", "srcn"], inplace=True)

    elif ptype == 'cycle':

        ### final tdf

        # list of all section labels (or names)
        keys = sorted(set(tdf["lsrc"].tolist() + tdf["ldest"].tolist()))
        values = list(range(0, len(keys)))
        m = dict(zip(keys, values))

        # assign an unique integer (id) to the labels
        tdf["src_id"] = tdf["lsrc"].map(m)
        tdf["dest_id"] = tdf["ldest"].map(m)

    else:

        print('EXIT: not correct "ptype" value')
        exit()

    print('tdf', tdf.shape, 'final - time:', round((time.time() - start_time) / 60, 2), 'min')

    return dfv, m, tdf


def res_sankey(tdf, ptype='linear'):
    ### nodes & links & names

    # nodes: map between an integer id and the section name
    d = {a: b for a, b in zip(list(tdf['src_id']) + list(tdf['dest_id']), list(tdf['lsrc']) + list(tdf['ldest']))}
    d = dict(sorted(d.items()))
    nodes = [{'node': e[0], 'name': e[1]} for e in d.items()]

    if ptype == 'linear':
        # labels: map between the unique journey code (step+code) and the section name
        labels = {a: b for a, b in zip(list(tdf['src']) + list(tdf['dest']), list(tdf['lsrc']) + list(tdf['ldest']))}
    elif ptype == 'cycle':
        # labels: dummy map between the section name and the section name
        labels = {a: b for a, b in zip(list(tdf['lsrc']) + list(tdf['ldest']), list(tdf['lsrc']) + list(tdf['ldest']))}
    labels = dict(sorted(labels.items()))
    labels.update({'ENTRY': 'ENTRY', 'EXIT': 'EXIT'})

    # links: all transitions, source -> target -> amount/value
    dfl = tdf[['src_id', 'dest_id', 'amount']].rename(
        columns={'src_id': 'source', 'dest_id': 'target', 'amount': 'value'})
    links = dfl.to_dict('records')

    res = {'nodes': nodes, 'links': links, 'labels': labels}
    assert len(res['nodes']) == res['nodes'][-1]['node'] + 1

    return res


def SankeyChart(tdf, res, m):

    sources = [r['source'] for r in res['links']]
    targets = [r['target'] for r in res['links']]
    values = [int(r['value']) for r in res['links']]
    labels = [res['labels'][k] for k in m.keys()]

    # colors
    colorlist = [colorize(item, "0.8") for item in labels]
    linkcolorlist = [colorize(item, "0.5") for item in tdf["lsrc"]]
    tdf["linkcolor"] = linkcolorlist
    tdf.loc[tdf['ldest'] == "ENTRY", "linkcolor"] = "rgba(128,128,128,0.0)"

    # nan_np = np.empty(len(tdf))
    # nan_np[:] = np.nan

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=30,
            thickness=5,
            line=dict(color="black", width=0.5),
            # x=np.array(tdf['srcl']),
            # y=nan_np,
            label=labels,
            color=colorlist
        ),
        arrangement='freeform',
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=tdf['linkcolor'],
            hovertemplate='%{value} from %{target.label} to %{source.label}.'
        )))
    fig.update_layout(title_text=(f"{int(tdf['amount'].sum())} transitions"),
                      font_size=10,
                      height=500)
    return fig



############################


def res_linear_bar(df_lp):
    dict_res, map_res = [], []
    for ii, i in enumerate(df_lp['JS_id'].unique()):
        tmp = df_lp.loc[df_lp['JS_id'] == i].copy()
        tmp['id'] = range(len(tmp))
        tmp['sid'] = tmp['Journey_S'] + '-' + tmp['id'].astype(str)
        # dict_res[i] = tmp.set_index('sid')['reltime_ts'].to_dict()
        dict_res += [{'MIDVN': i}]
        map_res += [{'MIDVN': i}]
        dict_res[ii].update(tmp.set_index('sid')['duration_ts'].to_dict())
        map_res[ii].update(tmp.set_index('sid')['Journey_S'].to_dict())

    return dict_res, map_res


def LinearBar(df_lp):

    df_lp['JS_id'] = df_lp['JS_id'].astype(str)

    df_lp.sort_values(by=['labels', 'last_touch_channel', 'device', 'visit_ts', 'JS_id', 'reltime_ts'], inplace=True)

    df_lp["JS_id_sorted"] = (df_lp['JS_id'] != df_lp['JS_id'].shift(1)).cumsum()
    df_lp["label_switch"] = df_lp['labels'].diff()
    df_lp['duration_ts_60'] = df_lp['duration_ts'] / 60

    df_lp['labels-'] = df_lp['labels'].astype(str) + ' '
    df_lp['device-'] = df_lp['device'].astype(str) + ' '
    df_lp['last_touch_channel-'] = df_lp['last_touch_channel'].astype(str) + ' '
    df_lp['yid'] = df_lp[['last_touch_channel-', 'device-', 'labels-', ]].astype(str).sum(1)

    fig = px.bar(df_lp, y="JS_id_sorted", x="duration_ts_60",
                 color='Journey_S', orientation='h', height=len(df_lp) * 1.5)
    for i in list(df_lp.loc[df_lp['label_switch']==1,'JS_id_sorted']):
        fig.add_hline(y=float(i) - .5, opacity=0.2)
    fig.update_layout(
        yaxis=dict(
            tickvals=df_lp['JS_id_sorted'].tolist(),
            ticktext=df_lp['yid'].tolist(),
        )
    )
    return fig



############################


def df_sankey_old(tmpj, start_date, end_date, section='None',
                  form_started='None', form_finished='None', p_NUM_MAXSTEPS=13):

    print (f"dates=({start_date, end_date}), section={section}, "
           f"form_started={form_started}, form_finished={form_finished}, p_NUM_MAXSTEPS={p_NUM_MAXSTEPS}")

    start_time = time.time()
    print('df:', tmpj.shape, tmpj['MIDVN'].nunique(), end=' -> ')

    ### filters

    tmpj['date'] = pd.to_datetime(tmpj['min_timestamp']).dt.date.astype(str)
    tmpj = tmpj.loc[(tmpj['date']>=start_date)&(tmpj['date']<=end_date)].copy()

    MIDVN_ref_list = set(tmpj['MIDVN'].unique())

    if form_started != 'None':
        MIDVN_ref_list_ = set(tmpj.loc[tmpj['form_started'].astype(str) == form_started, 'MIDVN'].unique())
        MIDVN_ref_list = MIDVN_ref_list.intersection(MIDVN_ref_list_)

    if form_finished != 'None':
        MIDVN_ref_list_ = set(tmpj.loc[tmpj['form_finished'].astype(str) == form_finished, 'MIDVN'].unique())
        MIDVN_ref_list = MIDVN_ref_list.intersection(MIDVN_ref_list_)

    if section != 'None':
        MIDVN_ref_list_ = list(set(tmpj.loc[tmpj['ref_point'].astype(str) == section, 'MIDVN'].unique()))
        section = f'_{section}_'
        MIDVN_ref_list_+= list(set(tmpj.loc[tmpj['sections'].astype(str).apply(lambda x: section in x), 'MIDVN'].unique()))
        MIDVN_ref_list = list(MIDVN_ref_list.intersection(MIDVN_ref_list_))

    tmpj = tmpj[tmpj.MIDVN.isin(MIDVN_ref_list)].copy()

    print(tmpj.shape, tmpj['MIDVN'].nunique())

    print('filters - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### jrn_ = pre_ + jcode_ (unique id)

    tmpj["jrn_00"] = tmpj.ref_point
    for i in range(1, p_NUM_MAXSTEPS + 1):
        tmpj["jrn_" + str(i).zfill(2)] = tmpj["pre_" + str(i).zfill(2)] + " ~ " + tmpj["jcode_" + str(i).zfill(2)]

    print('jrn_ - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### jrn_ & cnt_

    ll = []
    for i in range(0, p_NUM_MAXSTEPS):
        ll.append(list(np.where(tmpj["jrn_" + str(i).zfill(2)].notnull() &
                                tmpj["jrn_" + str(i + 1).zfill(2)].notnull(), \
                                tmpj["jrn_" + str(i).zfill(2)] + ">" +
                                tmpj["jrn_" + str(i + 1).zfill(2)] + ">" +
                                tmpj["cnt_" + str(i + 1).zfill(2)].astype(str), None)))

    # clean
    fll = [item for sublist in ll for item in sublist]
    fll = list(set(fll))
    fll = list(filter(None, fll))

    print('jrn_&cnt_ - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### transitions (with amount): source (src) > target (dest)

    tdf = pd.DataFrame(fll, columns=["transitions"])
    tdf["src"] = tdf["transitions"].str.split(pat=">").str[0]
    tdf["dest"] = tdf["transitions"].str.split(pat=">").str[1]
    tdf["amount"] = tdf["transitions"].str.split(pat=">").str[2]
    tdf["lsrc"] = tdf["src"].str.split(pat=" ~ ").str[0]
    tdf["ldest"] = tdf["dest"].str.split(pat=" ~ ").str[0]

    print('transitions - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### remove amounts <3

    p_paththr = 3
    tdf['amount'] = tdf['amount'].astype(float)
    tdf = tdf[tdf['amount'].ge(p_paththr)]

    print('remove<3 - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### not understood!!! dummy??

    dll = []
    for item in list(set(tdf.dest.values).difference(set(tdf.src.values))):
        dll.append([item + ">DUMMY>1.0", item, "DUMMY", 1.0, item.split(" ~ ")[0], "DUMMY"])

    # tdf = tdf.append(pd.DataFrame(dll, columns=tdf.columns))
    tdf = pd.concat([tdf, pd.DataFrame(dll, columns=tdf.columns)], ignore_index=True)

    print('not-understood - time:', round((time.time() - start_time) / 60, 2), 'min')

    ### final df

    keys = list(set(tdf["src"].unique()).union(set(tdf["dest"].unique())))
    values = list(range(0, len(keys)))
    m = dict(zip(keys, values))

    tdf["src_id"] = tdf["src"].map(m)
    tdf["dest_id"] = tdf["dest"].map(m)

    tdf["srcn"] = tdf["src"].str.split(pat="~").str[1]

    tdf['srcl'] = tdf['srcn'].str.len() / 10
    tdf.sort_values(by=["srcl", "srcn"], inplace=True)

    print('final-df - time:', round((time.time() - start_time) / 60, 2), 'min')

    return tmpj, tdf
