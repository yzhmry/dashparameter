#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=50)


import dash
import dash_table
import dash_core_components as dcc 
import dash_bootstrap_components as dbc                 # 交互式组件
import dash_html_components as html                 # 代码转html
from dash.dependencies import Input, Output, State,ClientsideFunction  # 回调
import pathlib

import base64
import datetime
import io
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)




#
#os.chdir(r'D:\ck\yzh\2 replenishment\8.modeling\dash_parameter')
#path1=r'D:\ck\yzh\2 replenishment\8.modeling\dash_parameter'
#
### Path
##BASE_PATH = pathlib.Path(__file__).parent.resolve()
#BASE_PATH = pathlib.Path(path1).resolve()
#DATA_PATH = BASE_PATH.joinpath("data").resolve()
#
#df_future=pd.read_csv(DATA_PATH.joinpath('store_replenishment_parameter_weekly_future.csv'),header=0,dtype='str',error_bad_lines=False)
#df_history=pd.read_csv(DATA_PATH.joinpath('holiday_parameter_weekly_history.csv'),header=0,dtype='str',error_bad_lines=False)
#p=pd.read_csv(DATA_PATH.joinpath('p.csv'),header=0,dtype='str',error_bad_lines=False)

####从数据库中获取数据
import psycopg2
DATABASE_URL = os.environ['DATABASE_URL']
#DATABASE_URL='postgres://rjrmrnsqlocmfa:e3d697ddc65c54d6331d0979776614c96b188a16e679ce88e0f953a1570c3091@ec2-50-19-26-235.compute-1.amazonaws.com:5432/de3l2qcsken8n'
conn = psycopg2.connect(DATABASE_URL, sslmode='require')
p=pd.read_sql("select * from p",conn)
df_history=pd.read_sql("select * from holiday_parameter_weekly_history",conn)
df_future=pd.read_sql("select * from store_replenishment_parameter_weekly_future",conn)

def next_wk(day):
    next_day=datetime.datetime.strptime(day,'%Y-%m-%d')-datetime.timedelta(days=21)
#     print(next_day)
    next_week=str(next_day.isocalendar()[0])+str(next_day.isocalendar()[1])
    return next_week

key_week=next_wk(datetime.date.today().strftime('%Y-%m-%d'))
##将多个sheet存入一张表)



p['week_ess']=p['start_week']+','+p['end_week']
p['week_ess']=[list(range(int(s.split(',')[0]),int(s.split(',')[1])+1)) for s in p['week_ess']]
week_ess=pd.DataFrame()
for i in p['week_ess']:
    l=pd.DataFrame(i)
    week_ess=pd.concat([week_ess,l])
week_ess.columns=['week']
week_ess['f']=week_ess['week'].astype(str).str[-2:].astype(int)
week_ess=week_ess[(week_ess['f']<=52) & (week_ess['f']>0)]
week_ess=week_ess.drop_duplicates()
print(len(week_ess))
week_ess['week']=week_ess['week'].astype(str)






##将历史数据和未来预测数据合并
df_history2=df_history[df_history['yw_lt']<key_week]
df_future1=df_future[df_future['yw_lt']>=key_week]

df_p=pd.concat([df_history2[[ 'level1','yw_ct', 'yw_lt',
       'yw_nt', 'yw_llt', 'holiday_lw', 'holiday_cw', 'holiday_nw', 'holiday_llw', 'parameter_year_avg', 'parameter_moving_avg_4w',
       'parameter_moving_avg_8w', 'parameter_moving_avg_13w']],df_future1[[ 'level1','yw_ct', 'yw_lt',
       'yw_nt', 'yw_llt', 'holiday_lw', 'holiday_cw', 'holiday_nw', 'holiday_llw', 'parameter_year_avg',
       'parameter_moving_avg_4w','parameter_moving_avg_8w', 'parameter_moving_avg_13w']]])

##########ESS周数合并
df_pp=pd.merge(df_p,week_ess,left_on='yw_lt',right_on='week',how='left',suffixes=('', '_lt'))
df_pp=pd.merge(df_pp,week_ess,left_on='yw_llt',right_on='week',how='left',suffixes=('', '_llt'))
df_pp=pd.merge(df_pp,week_ess,left_on='yw_ct',right_on='week',how='left',suffixes=('', '_ct'))
df_pp=pd.merge(df_pp,week_ess,left_on='yw_nt',right_on='week',how='left',suffixes=('', '_nt'))
df_pp.columns
df_pp=df_pp[[ 'level1','yw_lt', 'holiday_lw', 'holiday_cw',
       'holiday_nw', 'holiday_llw', 'parameter_year_avg',
       'parameter_moving_avg_4w', 'parameter_moving_avg_8w',
       'parameter_moving_avg_13w', 'week', 'week_llt',  'week_ct', 'week_nt' ]]
df_pp.columns=[ 'level1','yw_lt', 'holiday_lw', 'holiday_cw',
       'holiday_nw', 'holiday_llw', 'parameter_year_avg',
       'parameter_moving_avg_4w', 'parameter_moving_avg_8w',
       'parameter_moving_avg_13w', 'ess_lt', 'ess_llt','ess_ct','ess_nt']
df_pp['ess_lt']=[1 if len(str(x))>3  else 0 for x in df_pp['ess_lt']]
df_pp['ess_llt']=[1 if len(str(x))>3  else 0 for x in df_pp['ess_llt']]
df_pp['ess_ct']=[1 if len(str(x))>3  else 0 for x in df_pp['ess_ct']]
df_pp['ess_nt']=[1 if len(str(x))>3  else 0 for x in df_pp['ess_nt']]


df_pp['year_lt']=df_pp['yw_lt'].str[:4].astype(int)
df_pp['week_lt']=df_pp['yw_lt'].str[4:].astype(int)
df_pp[[ 'parameter_year_avg', 'parameter_moving_avg_4w',
       'parameter_moving_avg_8w', 'parameter_moving_avg_13w']]=df_pp[[ 'parameter_year_avg', 'parameter_moving_avg_4w',
       'parameter_moving_avg_8w', 'parameter_moving_avg_13w']].astype(float)

feature_select=['level1','year_lt', 'week_lt', 'holiday_lw', 'holiday_cw', 'holiday_nw', 'holiday_llw','ess_lt', 'ess_llt','ess_ct','ess_nt']

# feature_select=['year', 'week', 'holiday_lw', 'holiday_cw', 'holiday_nw', 'holiday_llw','cnt_2w']
yw_select=df_p['yw_lt'].unique()
yw_select0='202026'
yw_select0

df_pp=df_pp.fillna(1.0)

X=pd.get_dummies(df_pp[feature_select])
X['yw_lt']=df_pp['yw_lt']
X_train=X[X['yw_lt']<yw_select0]
del X_train['yw_lt']
X_n0=X[X['yw_lt']>=yw_select0]
del X_n0['yw_lt']
y_train=df_pp[df_pp['yw_lt']<yw_select0][[ 'parameter_year_avg', 'parameter_moving_avg_4w','parameter_moving_avg_8w', 'parameter_moving_avg_13w']]
y_n0=df_pp[df_pp['yw_lt']>=yw_select0][['parameter_year_avg','parameter_moving_avg_4w', 'parameter_moving_avg_8w', 'parameter_moving_avg_13w']]
df_n0=df_pp[df_pp['yw_lt']>=yw_select0]





model = ensemble.RandomForestRegressor(n_estimators =50)
model.fit(X_train,y_train)
y_pred = model.predict(X_train)
y_pred0 = model.predict(X_n0)
r2_train='train R2 决定系数（拟合优度）: %.2f' % r2_score(y_train,y_pred)
print('train R2 决定系数（拟合优度）: %.2f' % r2_score(y_train,y_pred))
# print('val R2 决定系数（拟合优度）: %.2f' % r2_score(y_n0,y_pred0))
print(np.mean(abs(y_n0-y_pred0)))
print(np.max(abs(y_n0-y_pred0)))
print(np.min(abs(y_n0-y_pred0)))


df_n0[['parameter_year_avg','parameter_moving_avg_4w', 'parameter_moving_avg_8w', 'parameter_moving_avg_13w']]=y_pred0

##合并模型数据与均值数据
df_pp0=pd.merge(df_future,df_n0[['parameter_year_avg','parameter_moving_avg_4w','parameter_moving_avg_8w', 'parameter_moving_avg_13w','yw_lt']],on='yw_lt',how='left')
df_pp0.columns

df_p0=df_pp0[['replenishment_day','level1','yw_lt', 'holiday_cw', 'holiday_nw','holiday_lw', 'holiday_llw',
       'parameter_manual', 'parameter_year_avg_x','parameter_moving_avg_4w_x', 'parameter_moving_avg_8w_x',
       'parameter_moving_avg_13w_x', 'parameter_year_avg_y','parameter_moving_avg_4w_y',
       'parameter_moving_avg_8w_y', 'parameter_moving_avg_13w_y']]



df_p0.columns=['replenishment_day','level1','yw_lt', 'holiday_cw', 'holiday_nw','holiday_lw', 'holiday_llw',
       'parameter_manual', 'parameter_year_avg', 'parameter_moving_avg_4w','parameter_moving_avg_8w',
       'parameter_moving_avg_13w', 'parameter_year_avg_model','parameter_moving_avg_4w_model',
       'parameter_moving_avg_8w_model', 'parameter_moving_avg_13w_model']




##显示1位小数
df_p0[[ 'parameter_manual', 'parameter_year_avg','parameter_moving_avg_4w', 'parameter_moving_avg_8w',
       'parameter_moving_avg_13w', 'parameter_year_avg_model','parameter_moving_avg_4w_model',
       'parameter_moving_avg_8w_model', 'parameter_moving_avg_13w_model']]=df_p0[[ 'parameter_manual', 
        'parameter_year_avg','parameter_moving_avg_4w', 'parameter_moving_avg_8w',
       'parameter_moving_avg_13w', 'parameter_year_avg_model','parameter_moving_avg_4w_model',
       'parameter_moving_avg_8w_model', 'parameter_moving_avg_13w_model']].astype(float)
df_p0[[ 'parameter_manual', 'parameter_year_avg','parameter_moving_avg_4w', 'parameter_moving_avg_8w',
       'parameter_moving_avg_13w', 'parameter_year_avg_model','parameter_moving_avg_4w_model',
       'parameter_moving_avg_8w_model', 'parameter_moving_avg_13w_model']]=df_p0[[ 'parameter_manual', 
        'parameter_year_avg','parameter_moving_avg_4w', 'parameter_moving_avg_8w',
       'parameter_moving_avg_13w', 'parameter_year_avg_model','parameter_moving_avg_4w_model',
       'parameter_moving_avg_8w_model', 'parameter_moving_avg_13w_model']].round(1)



df_p0=df_p0[['replenishment_day','level1','yw_lt', 'holiday_cw', 'holiday_nw','holiday_lw', 'holiday_llw',
             'parameter_manual', 'parameter_year_avg','parameter_moving_avg_4w', 'parameter_moving_avg_8w',
       'parameter_moving_avg_13w', 'parameter_moving_avg_4w_model',
       'parameter_moving_avg_8w_model', 'parameter_moving_avg_13w_model']]
#df_p0.dtypes

df_history[[  'parameter_year_avg','parameter_moving_avg_4w',  'parameter_moving_avg_8w', 'parameter_moving_avg_13w']]=df_history[[  'parameter_year_avg','parameter_moving_avg_4w',  'parameter_moving_avg_8w',
       'parameter_moving_avg_13w']].astype(float)

df_history[[  'parameter_year_avg','parameter_moving_avg_4w',  'parameter_moving_avg_8w', 'parameter_moving_avg_13w']]=df_history[[  'parameter_year_avg','parameter_moving_avg_4w',  'parameter_moving_avg_8w',
       'parameter_moving_avg_13w']].round(1)

df_g=df_p0
df_g['year']=df_p0['yw_lt'].str[:4]
df_g['week']=df_p0['yw_lt'].str[-2:]



# df_pp.to_csv(DATA_PATH.joinpath('p.csv'),index=False)



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # the Flask app
app.config.suppress_callback_exceptions = True



navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=app.get_asset_url("ck_logo.png"),height="30px")),
#                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("replenishment parameter", className="ml-2"),
                            style={'fontsize':24}),
                ],
                align="center",
                no_gutters=True,
            )
#            ,
#            href="https://www.charleskeith.cn/"
        )
    ],
    color="Black",
    dark=True,
    fixed='top'
)




def history_table(level1='SHOES'):
    """基于dataframe，设置表格格式"""
    table=dash_table.DataTable(
#        id='output_df',
        data=df_history[['level1','yw_lt','holiday_cw', 'holiday_nw','holiday_lw', 'holiday_llw',
        'parameter_year_avg', 'parameter_moving_avg_8w', 'parameter_moving_avg_13w']][df_history['level1']==level1].to_dict('records'),
                         
                         
        columns=[{'id': c, 'name': c} for c in ['level1','yw_lt','holiday_cw', 'holiday_nw','holiday_lw', 'holiday_llw',
        'parameter_year_avg', 'parameter_moving_avg_8w', 'parameter_moving_avg_13w']],
#         page_size=20,
         fixed_rows={'headers': True},
        style_table={
            'minWidth': '80%',
            'width':'1000px'
        },
        style_cell={
            'whiteSpace': 'normal',
            'textAlign': 'middle',
            'minWidth': 200, 'maxWidth': 200, 'width': 200
        },
        style_header={
            'backgroundColor': 'black',#底色
            'color':'white',#文字颜色
            'fontWeight': 'bold',
#             'height': 'auto',
            'overflow': 'display',
            },
        style_data_conditional=[
            { 'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
            }        
            ],
        editable=True,#可编辑
        export_format='xlsx',#可下载
        export_headers='display',
#         merge_duplicate_headers=True
#         ,
#         style_as_list_view=True#将垂直网格线去掉
         page_action='none'
        )
        
    return table


df_p0=df_p0.drop_duplicates()
def predict_table(level1='SHOES'):
    """基于dataframe，设置表格格式"""
    table=dash_table.DataTable(
#        id='output_df',
        data=df_p0[df_p0['level1']==level1].to_dict('records'),
        columns=[{'id': c, 'name': c} for c in df_p0.columns],
#         page_size=20,
       fixed_rows={'headers': True},
#         fixed_columns={'headers': True},
#        page_action='none',
        style_table={
            'minWidth': '80%',
            'width':'1000px'
#             'height': '1000px',
#                      'overflowY': 'auto'
        },
        style_cell={
#         whiteSpace': 'normal',
                    'textAlign': 'middle',
            'minWidth': 200, 'maxWidth': 200, 'width': 200
        },
#          style_data={
#         'whiteSpace': 'normal',
#         'height': 'auto',
#         'lineHeight': '15px'
#     },
        style_header={
            'backgroundColor': 'black',#底色
            'color':'white',#文字颜色
            'fontWeight': 'bold',
#             'height': 'auto',
            'overflow': 'display',
            },
        style_data_conditional=[
            { 'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
            },
            {'if': {
                    'filter_query': "{replenishment_day}>='2020-07-02'"
#                 ,
#                     'column_id': 'replenishment_day'
                },
                'color': 'tomato',
                'fontWeight': 'bold'
            },
             
            ],
#         virtualization=True,
        editable=True,#可编辑
        export_format='xlsx',#可下载
        export_headers='display'
#         merge_duplicate_headers=True
#         ,
#         style_as_list_view=True#将垂直网格线去掉
#          page_action='none'
        )
    return table





def line_graph1(level1='SHOES'):
    fig = px.line(df_history[df_history['level1']==level1], x="week", y="sold_lw", color='year',
                  title='history sales trend weekly-%s'%level1)
    fig.update_layout(
        width=1600,
        height=400)
    return dcc.Graph(id='line1',figure=fig)


def heatmap_graph1(level1='SHOES'): 
    df_history['sold_lw']= df_history['sold_lw'].astype(int)   
    df_hisp=df_history[df_history['level1']==level1].pivot_table(index=['week'],columns=['year'],values=['sold_lw'],
                   aggfunc=[np.sum],fill_value=0)    
    ##计算环比    
    df_his_per=df_hisp.pct_change(periods=1) 
    ##数据框+1后取整
    df_hisf=df_his_per.apply(lambda x:round(x+1,1))
    df_hisf=df_hisf.fillna(0)
    df_hisf.columns=df_history['year'].astype(str).unique().tolist()
    print(df_hisf.columns)
    df_hisf['2015']=[0.0 if str(x)=='inf' else x for x in df_hisf['2015']]
    fig = ff.create_annotated_heatmap(z=np.array(df_hisf.T),x=df_hisf.T.columns.tolist(),y=df_hisf.T.index.tolist(), 
                                      colorscale='Blues')
    fig.update_layout(
        width=1560,
        height=400,
        showlegend=False,
        title_text="历年环比-parameter-%s"%level1)
    return dcc.Graph(id='heatmap1',figure=fig)

def heatmap_graph_manual(level1='SHOES'): 
    df_gp=df_g[(df_g['level1']==level1)&(df_g['year']<='2020')].pivot_table(index=['week'],columns=['year'],values=['parameter_manual'],
                   aggfunc=[np.mean],fill_value=0)  
    df_gp.columns=df_g['year'][(df_g['level1']==level1)&(df_g['year']<='2020')].unique().tolist()
    fig = ff.create_annotated_heatmap(z=np.array(df_gp.T),x=df_gp.T.columns.tolist(),y=df_gp.T.index.tolist(), 
                                      colorscale='Blues')
    fig.update_layout(
        width=1560,
        height=300,
        showlegend=False,
        title_text="目前所用的parameter_manual-%s"%level1)
    return dcc.Graph(id='heatmap_manual',figure=fig)



def heatmap_graph_avg(level1='SHOES',parameter='parameter_moving_avg_4w'): 
    df_gp=df_g[(df_g['level1']==level1)&(df_g['year']<='2020')].pivot_table(index=['week'],columns=['year'],values=[parameter],
                   aggfunc=[np.mean],fill_value=0)  
    df_gp.columns=df_g['year'][(df_g['level1']==level1)&(df_g['year']<='2020')].unique().tolist()
    fig = ff.create_annotated_heatmap(z=np.array(df_gp.T),x=df_gp.T.columns.tolist(),y=df_gp.T.index.tolist(), 
                                      colorscale='Blues')
    
    fig.update_layout(
        width=1560,
        height=300,
        showlegend=False,
        title_text="%s-%s"%(parameter,level1))
    return dcc.Graph(id='heatmap_avg_4w',figure=fig)
df_g=df_g.fillna(0.0)

def heatmap_graph_model(level1='SHOES',parameter='parameter_moving_avg_4w_model'): 
    df_gp=df_g[(df_g['level1']==level1)&(df_g['year']<='2020')].pivot_table(index=['week'],columns=['year'],values=[parameter],
                   aggfunc=[np.mean],fill_value=0)  
    df_gp.columns=df_g['year'][(df_g['level1']==level1)&(df_g['year']<='2020')].unique().tolist()
    fig = ff.create_annotated_heatmap(z=np.array(df_gp.T),x=df_gp.T.columns.tolist(),y=df_gp.T.index.tolist(), 
                                      colorscale='Blues')
    fig.update_layout(
        width=1560,
        height=300,
        showlegend=False,
        title_text="%s-%s"%(parameter,level1))
    return dcc.Graph(id='heatmap_model_4w',figure=fig)

#
#
#df_gp=df_g.pivot_table(index=['year'],columns=['week'],values=['parameter_manual'],
#                   aggfunc=[np.mean],fill_value=0)  
#df_gp_4w_avg=df_g.pivot_table(index=['year'],columns=['week'],values=['parameter_moving_avg_4w'],
#                   aggfunc=[np.mean],fill_value=0)  
#df_gp_4w_model=df_g.pivot_table(index=['year'],columns=['week'],values=['parameter_moving_avg_4w_model'],
#                   aggfunc=[np.mean],fill_value=0)
#df_gp=df_gp.astype(float)


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return [
#                        html.H5("Replenishment Model Analysis"),
                        html.Label("Select level1"),
                        dcc.Dropdown(
                            id="level1-select",
                            options=[{"label": i, "value": i} for i in ['BAGS','SHOES']],
                            value='SHOES'
                        ),
                        html.Br(),
                        html.Label('Select moving_avg week'),
                        dcc.Dropdown(
                            id="ma-select",
                            options = [{"label": i, "value": i} for i in ['parameter_year_avg',
                               'parameter_moving_avg_4w', 'parameter_moving_avg_8w',
                               'parameter_moving_avg_13w']],
                            value='parameter_moving_avg_4w'
#                            value = feature_select[:],
#                            multi = True
                            ),
                        html.Br(),
                        html.Label('Select model week'),
                        dcc.Dropdown(
                            id="model-select",
                            options = [{"label": i, "value": i} for i in ['parameter_moving_avg_4w_model',
                                     'parameter_moving_avg_8w_model', 'parameter_moving_avg_13w_model']],
                            value='parameter_moving_avg_4w_model'
#                            value = feature_select[:],
#                            multi = True
                            ),
                        html.Br()
                        ]
#  "Blackbody", "Bluered", "Blues", "Earth", "Electric", "Greens", "Greys", "Hot", "Jet", 
#  "Picnic", "Portland", "Rainbow", "RdBu", "Reds", "Viridis", "YlGnBu", "YlOrRd".  
    
def display_tab(level1,ma_parameter,model_parameter):
    return  dcc.Tab(label='display',id='display', children=[
                   line_graph1(level1),
                   heatmap_graph1(level1),
                   heatmap_graph_manual(level1),
                   heatmap_graph_avg(level1,ma_parameter),
                   heatmap_graph_model(level1,model_parameter)
                   ])
        
    
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app.layout = html.Div(
    className="",    
    children = [
            navbar,
             html.Div(
            id="control-card",
            children=generate_control_card()
            ,
            style={
                 'width': '300px',
                 'height':'100px',
                 'border': '5px Aquamarine',
                 'margin-top': '80px',
                 'position':'relative'
                 }
            ),       
#         
         html.Div(
        dcc.Tabs([
            display_tab(level1='SHOES',ma_parameter='parameter_moving_avg_4w',
                                 model_parameter='parameter_moving_avg_4w_model'),
            dcc.Tab(label='predict parameter',id = 'predict_table',
                    children=[
                
                    dcc.Markdown('''
                    - sold_2w:未来两周销售
                    - sold_noholiday:历史两周销售中没有节假日的平均值
                    - parameter=sold_2w/sold_noholiday
                    - 红字代表预测值，黑字代表历史数据
                    ''') ,
                   predict_table(level1='SHOES')
                    ]),            
            dcc.Tab(label='history parameter',id = 'history_table',
                    children=[
                   history_table(level1='SHOES')
         
                    ])
#    ,

#            dcc.Tab(label='display', children=[
#                   line_graph1(level1='SHOES'),
#                   heatmap_graph1(level1='SHOES')
#                   ])
#                   ,
#            dcc.Tab(label='display-bags', children=[
#                   line_graph2(level1='BAGS'),
#                   heatmap_graph2(level1='BAGS')
#                   ])
            
#        dcc.Tab(label='predict parameter2', children=[
#            dbc.Table.from_dataframe(df_pp, striped=True, bordered=True, hover=True)
#         ])    

    ])
                             ,
            style={
#                 'width': '200px',
#                 'height':'100px',
#                 'border': '5px Aquamarine',
                 'margin-top': '-100px',
                 'margin-left':'300px'
                 })
    ]
    ,
        style= {'margin-left': '20px'})
                


#原数据的显示
@app.callback(
    [Output("history_table", "children"),
     Output("predict_table", "children"),
     Output("display", "children")
#     Output("line1", "figure"),
#     Output("heatmap1", "figure")
    ]
    ,
    [Input("level1-select", "value"),Input("ma-select", "value"),Input("model-select", "value")])
    
def update_line(level1,ma_parameter,model_parameter):
#    return [history_table(level1),predict_table(level1),line_graph1(level1),heatmap_graph1(level1)]
    return [history_table(level1),predict_table(level1),display_tab(level1,ma_parameter,model_parameter)]






if __name__ == '__main__':
    app.run_server(        
#         port=8050,
#         host='0.0.0.0'
    )


