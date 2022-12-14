# %%

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# %%
h=pd.read_csv("data2.csv",index_col='Id')
x=pd.read_csv('data.csv')
cs=pd.read_csv('company_score.csv',index_col='Id')
st.header("FleetRisk Prediction Website")
st.write("An Efficient Platform to evaluate Driver,Route and Vehicle Performance to ensure fleet safety!")
dict={'DSP1':x['DSP1'][0],'DSP2':x['DSP2'][0],'DSP3':x['DSP3'][0],'n1':x['n1'][0],'n2':x['n2'][0],'n3':x['n3'][0]}
dict2={'DSP1':x['DSP1'][0],'DSP2':x['DSP2'][0],'DSP3':x['DSP3'][0]}
# %%

st.markdown(
         f"""
         <style>
         *{{
            font-weight:bold;
            font-height: 1.23rem;
         }}
         .css-10xlvwk {{
            background-color: rgb(45 55 238 / 50%);
         }}
         .css-18ni7ap {{
            height : 0rem;
         }}
        #  .stApp {{
        #      background-image: url("data:image/jpeg;base64,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");
        #      background-attachment: fixed;
        #      background-size: cover
        #  }}
         .stAlert{{
            background-color:rgb(71 214 115 / 80%);
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
# %%

with st.sidebar:
    
    st.title("Welcome to fleetrisk prediction website!")
source = ("Select from below ","Driver Risk Score", "Route Risk Score","Vehicle Risk Score","DSP score","Individual DSP Analysis","DSP comparision","Company Score and Analysis")
source_index = st.sidebar.selectbox("Select:", range(
        len(source)), format_func=lambda x: source[x])

# %%
if source_index == 0:
    with st.container():
        st.video("https://media.istockphoto.com/id/1334409221/video/gameplay-of-a-racing-simulator-video-game-with-interface-computer-generated-3d-car-driving.mp4?s=mp4-640x640-is&k=20&c=KNi0JAmT-lP2J1xzNEZUfad6k0BfiHBHh26C_xd_LSQ=",start_time=0)

if source_index == 1:
    st.header("Driver Risk Score")
    col1, col2, col3 = st.columns(3)
    with col1:
        DRIVER_ID = st.number_input("Enter Driver Id",value=0)
        DSP_NO = st.number_input("Enter DSP number",value=0)
        DRIVER_NAME=st.text_input("Enter Driver Name")
        AGE = st.number_input("Enter Age",value=0)
        LICENSE_DETAILS=st.text_input("Enter License Details")
        NUMBER_OF_TRIPS=st.number_input("Enter NUMBER_OF_TRIPS",value=0)
    with col2:  
        REWARD_POINTS=st.number_input("Enter REWARD_POINTS",value=0)
        TOTAL_MILES_DONE= st.number_input("Enter Total miles done",value=0)
        MILES_IN_URBAN= st.number_input("Enter miles done in urban",value=0)
        MILES_IN_NIGHT= st.number_input("Enter MILES_IN_NIGHT",value=0)
        MILES_DONE_IN_RURAL= st.number_input("Enter MILES DONE IN RURAL",value=0)
        ACCELERATION= st.number_input("Enter ACCELERATION",value=0)
    with col3:
        BRAKING= st.number_input("Enter BRAKING",value=0)
        CORNERING= st.number_input("Enter CORNERING",value=0)
        SPEEDING= st.number_input("Enter SPEEDING",value=0)
        SEATBELT= st.number_input("Enter SEATBELT",value=0)
        DISTRACTION= st.number_input("Enter DISTRACTION",value=0)
        NUMBER_OF_TICKETS_RECEIVED= st.number_input("Enter NO_OF_TICKETS_RECEIVED",value=0)


    if st.button("Submit"):
    
    # Unpickle classifier
        model_gs = joblib.load("model_gs2.pkl")
    
    # Store inputs into dataframe
        X = pd.DataFrame([[AGE,NUMBER_OF_TRIPS,REWARD_POINTS,TOTAL_MILES_DONE,MILES_IN_URBAN,MILES_IN_NIGHT,MILES_DONE_IN_RURAL,ACCELERATION,BRAKING,CORNERING,SPEEDING,SEATBELT,DISTRACTION,NUMBER_OF_TICKETS_RECEIVED]], 
                     columns = ["AGE","NUMBER_OF_TRIPS","REWARD_POINTS", "TOTAL_MILES_DONE", "MILES_IN_URBAN", "MILES_IN_NIGHT", "MILES DONE IN RURAL", "ACCELERATION", "BRAKING", "CORNERING", "SPEEDING", "SEATBELT", "DISTRACTION", "NUMBER_OF_TICKETS_RECEIVED"])
    # X = X.replace(["Brown", "Blue"], [1, 0])
    # ,DRIVER_ID,AGE,NUMBER_OF_TRIPS,SAFETY_SCORE,REWARD_POINTS,TOTAL_MILES_DONE,MILES_IN_URBAN,MILES_IN_NIGHT,MILES DONE IN RURAL,ACCELERATION,BRAKING,CORNERING,SPEEDING,SEATBELT,DISTRACTION,NUMBER_OF_TICKETS_RECEIVED

    # Get prediction
        prediction = model_gs.predict(X)[0]
        prediction=int(prediction)
        if DSP_NO == 1:
            dsp1=pd.read_csv("dsp1.csv")
            df2 = {'DRIVER_ID': DRIVER_ID, 'AGE': AGE, 'NUMBER_OF_TRIPS': NUMBER_OF_TRIPS,'SAFETY_SCORE': prediction,'REWARD_POINTS': REWARD_POINTS,'TOTAL_MILES_DONE': TOTAL_MILES_DONE,'MILES_IN_URBAN': MILES_IN_URBAN,'MILES_IN_NIGHT': MILES_IN_NIGHT,'MILES DONE IN RURAL': MILES_DONE_IN_RURAL,'ACCELERATION': ACCELERATION,'BRAKING': BRAKING,'CORNERING': CORNERING,'SPEEDING': SPEEDING,'SEATBELT': SEATBELT,'DISTRACTION': DISTRACTION,'NUMBER_OF_TICKETS_RECEIVED': NUMBER_OF_TICKETS_RECEIVED,}
            dsp1 = dsp1.append(df2, ignore_index = True)
            dsp1=dsp1.drop(dsp1.columns[[0]],axis=1)
            dsp1.to_csv("dsp1.csv")
            # new = pd.DataFrame(df2, index=[0])
            # # new = pd.DataFrame.from_dict(df2,index='DRIVER_ID')
            # new.set_index("DRIVER_ID")
            # dsp11= pd.concat([dsp1, new])
            # dsp11.reset_index()
            # dsp11
            n1=dict['n1']
            x=dict['DSP1']
            dict['DSP1']=((n1*x)+prediction)/(n1+1)
            # h["Score"][0]=dict['DSP1']
            h.iloc[0,1]=dict['DSP1']
            dict['n1']=n1+1
        if DSP_NO == 2:
            dsp2=pd.read_csv("dsp2.csv")
            df2 = {'DRIVER_ID': DRIVER_ID, 'AGE': AGE, 'NUMBER_OF_TRIPS': NUMBER_OF_TRIPS,'SAFETY_SCORE': prediction,'REWARD_POINTS': REWARD_POINTS,'TOTAL_MILES_DONE': TOTAL_MILES_DONE,'MILES_IN_URBAN': MILES_IN_URBAN,'MILES_IN_NIGHT': MILES_IN_NIGHT,'MILES DONE IN RURAL': MILES_DONE_IN_RURAL,'ACCELERATION': ACCELERATION,'BRAKING': BRAKING,'CORNERING': CORNERING,'SPEEDING': SPEEDING,'SEATBELT': SEATBELT,'DISTRACTION': DISTRACTION,'NUMBER_OF_TICKETS_RECEIVED': NUMBER_OF_TICKETS_RECEIVED,}
            dsp2 = dsp2.append(df2, ignore_index = True)
            dsp2=dsp2.drop(dsp2.columns[[0]],axis=1)
            dsp2.to_csv("dsp2.csv")
            n2=dict['n2']
            x=dict['DSP2']
            dict['DSP2']=((n2*x)+prediction)/(n2+1)
            # h["Score"][1]=dict['DSP2']
            h.iloc[1,1]=dict['DSP2']
            dict['n2']=n2+1
        if DSP_NO == 3:
            dsp3=pd.read_csv("dsp1.csv")
            df2 = {'DRIVER_ID': DRIVER_ID, 'AGE': AGE, 'NUMBER_OF_TRIPS': NUMBER_OF_TRIPS,'SAFETY_SCORE': prediction,'REWARD_POINTS': REWARD_POINTS,'TOTAL_MILES_DONE': TOTAL_MILES_DONE,'MILES_IN_URBAN': MILES_IN_URBAN,'MILES_IN_NIGHT': MILES_IN_NIGHT,'MILES DONE IN RURAL': MILES_DONE_IN_RURAL,'ACCELERATION': ACCELERATION,'BRAKING': BRAKING,'CORNERING': CORNERING,'SPEEDING': SPEEDING,'SEATBELT': SEATBELT,'DISTRACTION': DISTRACTION,'NUMBER_OF_TICKETS_RECEIVED': NUMBER_OF_TICKETS_RECEIVED,}
            dsp3 = dsp3.append(df2, ignore_index = True)
            dsp3=dsp3.drop(dsp3.columns[[0]],axis=1)
            dsp3.to_csv("dsp3.csv")
            n3=dict['n3']
            x=dict['DSP3']
            dict['DSP3']=((n3*x)+prediction)/(n3+1)
            # h["Score"][2]=dict['DSP3']
            h.iloc[2,1]=dict['DSP3']
            dict['n3']=n3+1
    # Output prediction
        import datetime
        st.success(f'Safety score : {prediction}%')
        cs=pd.read_csv("company_score.csv")
        from datetime import date
        from datetime import datetime
        # Returns the current local date
        today = date.today()

        now = datetime.now()
        # date=datetime.date.today()
        year = today.strftime("%Y")
        res = now.strftime("%H:%M:%S")
        # cs.reset_index()
        # df2 = { 'TIME': res, 'Score': np.average([dict['DSP1'],dict['DSP2'],dict['DSP3']])}
        # cs = cs.append(df2,ignore_index=True)
        l=len(cs.index)
        df1 = pd.DataFrame({"TIME": [res],"Score": [np.average([dict['DSP1'],dict['DSP2'],dict['DSP3']])],"Id":[l],"DATE":[today],"YEAR":[year]})
        df1.set_index("Id")
        df1.index.name = "Id"
        # cs.loc[f"{len(cs.index)}"] = [res, np.average([dict['DSP1'],dict['DSP2'],dict['DSP3']])] 
        
        df2 = pd.concat([cs, df1])
        
        # cs.set_index('Id')
        # st.text(f"The risk score of driver is {prediction}")
        x = pd.DataFrame([[dict['DSP1'],dict['DSP2'],dict['DSP3'],dict['n1'],dict['n2'],dict['n3']]], 
                     columns = ["DSP1","DSP2","DSP3","n1","n2","n3"])
        x.to_csv("data.csv")
        h.to_csv("data2.csv")
        df2.to_csv("company_score.csv",index=False)


# %%
# DESTINATIONS	RAIN	TEMP	PRESSURE	WIND_SPEED	WIND_DIRECTION
if source_index==2:
    st.header("Route Risk Score")
    Col1, Col2 = st.columns(2)
    with Col1:
        ROUTE_ID=st.number_input("Enter ROUTE_ID",value=0)
        DESTINATIONS = st.number_input("Enter DESTINATIONS",value=0)
        RAIN = st.number_input("Enter RAIN",value=0)
        TEMP = st.number_input("Enter TEMP",value=0)
    with Col2:
        PRESSURE = st.number_input("Enter PRESSURE",value=0)
        WIND_SPEED = st.number_input("Enter WIND_SPEED",value=0)
        WIND_DIRECTION = st.number_input("Enter WIND_DIRECTION",value=0)
    

    if st.button("Submit"):
    
    # Unpickle classifier
        model_route = joblib.load("pipeline_route1.pkl")
    
    # Store inputs into dataframe
        X = pd.DataFrame([[DESTINATIONS,RAIN,TEMP,PRESSURE,WIND_SPEED,WIND_DIRECTION]], 
                     columns = ["DESTINATIONS", "RAIN", "TEMP", "PRESSURE", "WIND_SPEED", "WIND_DIRECTION"])
    # X = X.replace(["Brown", "Blue"], [1, 0])
    
    # Get prediction
        prediction1 = model_route.predict(X)[0]
        pred = int(prediction1)
        
        st.success(f'Risk score : {pred}%')
# %%
if source_index==3:
    st.header("Vehicle Risk Score")
    COL1, COL2, COL3 = st.columns(3)
    with COL1:
        YEAR = st.number_input("Enter YEAR",value=0)
        TOTAL_MILES_DONE = st.number_input("Enter TOTAL_MILES_DONE",value=0)
        BATTERY_VOLTAGE = st.number_input("Enter BATTERY_VOLTAGE",value=0)
        TYRE_PRESSURE = st.number_input("Enter TYRE_PRESSURE",value=0)
    with COL2:
        FUEL_LEVEL = st.number_input("Enter FUEL_LEVEL",value=0)
        OIL_LEVEL = st.number_input("Enter OIL_LEVEL",value=0)
        DASH_CAM_IP = st.number_input("Enter DASH_CAM_IP",value=0)
    with COL3:
        LAST_SERVICE_DATE = st.number_input("Enter LAST_SERVICE_DATE",value=0)
        NEXT_SERVICE_DATE = st.number_input("Enter NEXT_SERVICE_DATE",value=0)
        NEXT_SERVICE_MILES = st.number_input("Enter NEXT_SERVICE_MILES",value=0)
    if st.button("Submit"):
    
    # Unpickle classifier
        model_veh = joblib.load("pipeline_vehicle1.pkl")
    
    # Store inputs into dataframe
        X = pd.DataFrame([[YEAR,TOTAL_MILES_DONE,BATTERY_VOLTAGE,TYRE_PRESSURE,FUEL_LEVEL,OIL_LEVEL,DASH_CAM_IP,LAST_SERVICE_DATE,NEXT_SERVICE_DATE,NEXT_SERVICE_MILES]], 
                     columns = ["YEAR", "TOTAL_MILES_DONE", "BATTERY_VOLTAGE", "TYRE_PRESSURE", "FUEL_LEVEL", "OIL_LEVEL","DASH_CAM_IP","LAST_SERVICE_DATE","NEXT_SERVICE_DATE","NEXT_SERVICE_MILES"])
    # X = X.replace(["Brown", "Blue"], [1, 0])
    
    # Get prediction
        prediction2 = model_veh.predict(X)[0]
        pred2 = int(prediction2)
        
        st.success(f"Safety score : {pred2}%")
# 	TOTAL_MILES_DONE		
# %%

if source_index==4:
    st.header("DSP Score")
    st.markdown(
         f"""
         <style>
         *{{
            font-weight:bold;
            font-height: 1.23rem;
         }}
         .css-znku1x p{{
            font-weight:bold;
            font-height: 1.23rem;
         }}
        #  .st-dy {{
        #     background-color: rgb(9 171 59 / 100%);
        # }}
        .css-znku1x e16nr0p33{{
            font-weight:bold;
            font-height: 1.23rem;
        }}
         </style>
         """,
         unsafe_allow_html=True
     )
    st.write("Enter the driver details in driver risk score section to update the dsp score")
    src=("DSP1","DSP2","DSP3")
    src_index = st.selectbox("Select:", range(
    len(src)), format_func=lambda x: src[x])
    if src_index==0:
        st.markdown(
         f"""
         <style>
         *{{
            font-weight:bold;
            font-height: 1.23rem;
         }}
         
        # .stAlert{{
        #     background-color: rgb(71 214 115 / 80%);
        # }}
        .
         </style>
         """,
         unsafe_allow_html=True
     )
        x=np.round(dict['DSP1'],2)
        st.success(f"DSP1 score: {x}%")
        
    if src_index==1:
        st.markdown(
         f"""
         <style>
         *{{
            font-weight:bold;
            font-height: 1.23rem;
         }}
         
        # .stAlert{{
        #     background-color: rgb(71 214 115 / 80%);
        # }}
        .
         </style>
         """,
         unsafe_allow_html=True
     )
        x=np.round(dict['DSP2'],2)
        st.success(f"DSP2 score: {x}%")
    if src_index==2:
        st.markdown(
         f"""
         <style>
         *{{
            font-weight:bold;
            font-height: 1.23rem;
         }}
         
        # .stAlert{{
        #     background-color: rgb(71 214 115 / 80%);
        # }}
        .
         </style>
         """,
         unsafe_allow_html=True
     )
        x=np.round(dict['DSP3'],2)
        st.success(f"DSP3 score: {x}%")
    Keymax = max(zip(dict2.values(), dict2.keys()))[1]
    st.write(f" {Keymax} is the safest DSP currently!")
    
if source_index==5:
    st.header("Individual DSP Analysis")
    tab1, tab2, tab3 = st.tabs(["DSP1", "DSP2", "DSP3"])
    with tab1:
        h1,h2=st.columns(2)
        with h1:
            st.write("DSP1:distribution of drivers w.r.t safety score")
            p1=pd.read_csv("dsp1.csv")
            c1=0
            c2=0
            c3=0
            c4=0
            c5=0
            for i,j in enumerate(p1["SAFETY_SCORE"]):
                if j>0 and j<20:
                    c1=c1+1
                if j>=20 and j<40:
                    c2=c2+1
                if j>=40 and j<60:
                    c3=c3+1
                if j>=60 and j<80:
                    c4=c4+1
                if j>=80 and j<100:
                    c5=c5+1
            data = {
            'Score': ['0-20', '20-40', '40-60', '60-80', '80-100'],
            'Count of drivers': [c1,c2,c3,c4,c5],}
            df = pd.DataFrame(data)
            df.set_index("Score",inplace=True)
            st.bar_chart(df)
        with h2:
            st.write("DSP1:distribution of drivers w.r.t no of violations")
            p1=pd.read_csv("dsp1.csv")
            c1=0
            c2=0
            c3=0
            c4=0
            c5=0
            for i,j in enumerate(p1["NUMBER_OF_TICKETS_RECEIVED"]):
                if j>0 and j<10:
                    c1=c1+1
                if j>=10 and j<20:
                    c2=c2+1
                if j>=20 and j<30:
                    c3=c3+1
                if j>=30 and j<40:
                    c4=c4+1
                if j>=40 and j<50:
                    c5=c5+1
            data = {
            'No of Violations': ['0-10', '10-20', '20-30', '30-40', '40-50'],
            'Count of drivers': [c1,c2,c3,c4,c5],}
            df = pd.DataFrame(data)
            df.set_index("No of Violations",inplace=True)
            st.bar_chart(df)
    with tab2:
        h1,h2=st.columns(2)
        with h1:
            st.write("DSP2:distribution of drivers w.r.t safety score")
            p1=pd.read_csv("dsp2.csv")
            c1=0
            c2=0
            c3=0
            c4=0
            c5=0
            for i,j in enumerate(p1["SAFETY_SCORE"]):
                if j>0 and j<20:
                    c1=c1+1
                if j>=20 and j<40:
                    c2=c2+1
                if j>=40 and j<60:
                    c3=c3+1
                if j>=60 and j<80:
                    c4=c4+1
                if j>=80 and j<100:
                    c5=c5+1
            data = {
            'Score': ['0-20', '20-40', '40-60', '60-80', '80-100'],
            'Count of drivers': [c1,c2,c3,c4,c5],}
            df = pd.DataFrame(data)
            df.set_index("Score",inplace=True)
            st.bar_chart(df)
        with h2:
            st.write("DSP2:distribution of drivers w.r.t no of violations")
            p1=pd.read_csv("dsp2.csv")
            c1=0
            c2=0
            c3=0
            c4=0
            c5=0
            for i,j in enumerate(p1["NUMBER_OF_TICKETS_RECEIVED"]):
                if j>0 and j<10:
                    c1=c1+1
                if j>=10 and j<20:
                    c2=c2+1
                if j>=20 and j<30:
                    c3=c3+1
                if j>=30 and j<40:
                    c4=c4+1
                if j>=40 and j<50:
                    c5=c5+1
            data = {
            'No of Violations': ['0-10', '10-20', '20-30', '30-40', '40-50'],
            'Count of drivers': [c1,c2,c3,c4,c5],}
            df = pd.DataFrame(data)
            df.set_index("No of Violations",inplace=True)
            st.bar_chart(df)
        
    with tab3:
        h1,h2=st.columns(2)
        with h1:
            st.write("DSP3:distribution of drivers w.r.t safety score")
            p1=pd.read_csv("dsp3.csv")
            c1=0
            c2=0
            c3=0
            c4=0
            c5=0
            for i,j in enumerate(p1["SAFETY_SCORE"]):
                if j>0 and j<20:
                    c1=c1+1
                if j>=20 and j<40:
                    c2=c2+1
                if j>=40 and j<60:
                    c3=c3+1
                if j>=60 and j<80:
                    c4=c4+1
                if j>=80 and j<100:
                    c5=c5+1
            data = {
            'Score': ['0-20', '20-40', '40-60', '60-80', '80-100'],
            'Count of drivers': [c1,c2,c3,c4,c5],}
            df = pd.DataFrame(data)
            df.set_index("Score",inplace=True)
            st.bar_chart(df)
        with h2:
            st.write("DSP3:distribution of drivers w.r.t no of violations")
            p1=pd.read_csv("dsp3.csv")
            c1=0
            c2=0
            c3=0
            c4=0
            c5=0
            for i,j in enumerate(p1["NUMBER_OF_TICKETS_RECEIVED"]):
                if j>0 and j<10:
                    c1=c1+1
                if j>=10 and j<20:
                    c2=c2+1
                if j>=20 and j<30:
                    c3=c3+1
                if j>=30 and j<40:
                    c4=c4+1
                if j>=40 and j<50:
                    c5=c5+1
            data = {
            'No of Violations': ['0-10', '10-20', '20-30', '30-40', '40-50'],
            'Count of drivers': [c1,c2,c3,c4,c5],}
            df = pd.DataFrame(data)
            df.set_index("No of Violations",inplace=True)
            st.bar_chart(df)

if source_index==6:
    st.header("DSP Comparison")
    h.set_index("DSP",inplace=True)
    st.bar_chart(h)
if source_index==7:
    tab1, tab2 = st.tabs(["Company Score", "Overall Analysis"])
    with tab1:
        dfc=pd.DataFrame()
        dfc["TIME"]=cs["TIME"]
        dfc["Score"]=cs["Score"]
        dfc.set_index("TIME",inplace=True)
        st.line_chart(dfc)
    with tab2:
        st.header("Analysis")

        p1=pd.read_csv("dsp1.csv")
        p2=pd.read_csv("dsp2.csv")
        p3=pd.read_csv("dsp3.csv")
        
        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i,j in enumerate(p1["SAFETY_SCORE"]):
            if j>0 and j<20:
                c1=c1+1
            if j>=20 and j<40:
                c2=c2+1
            if j>=40 and j<60:
                c3=c3+1
            if j>=60 and j<80:
                c4=c4+1
            if j>=80 and j<100:
                c5=c5+1
        for i,j in enumerate(p2["SAFETY_SCORE"]):
            if j>0 and j<20:
                c1=c1+1
            if j>=20 and j<40:
                c2=c2+1
            if j>=40 and j<60:
                c3=c3+1
            if j>=60 and j<80:
                c4=c4+1
            if j>=80 and j<100:
                c5=c5+1
        for i,j in enumerate(p3["SAFETY_SCORE"]):
            if j>0 and j<20:
                c1=c1+1
            if j>=20 and j<40:
                c2=c2+1
            if j>=40 and j<60:
                c3=c3+1
            if j>=60 and j<80:
                c4=c4+1
            if j>=80 and j<100:
                c5=c5+1
        data = {
        'Score': ['0-20', '20-40', '40-60', '60-80', '80-100'],
        'Count of drivers': [c1,c2,c3,c4,c5],}
        df = pd.DataFrame(data)
        df.set_index("Score",inplace=True)
        st.bar_chart(df)    

    

