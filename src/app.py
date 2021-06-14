import pickle
import streamlit as st
from streamlit import caching

import SessionState
# Assuming SessionState.py lives on this folder

session = SessionState.get(run_id=0)

pickle_in = open('classifier.pkl', 'rb')

classifier = pickle.load(pickle_in)


@st.cache(allow_output_mutation=True)
def prediction(Gender, Feature_2, Feature_3, Feature_4, Feature_5, Feature_28, Feature_29, Feature_37, Feature_39,
               Feature_40, Feature_41, Feature_43, Feature_50):
    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0

    if Feature_28 == "Every Day":
        Feature_28 = 0
    elif Feature_28 == "1-2 Days a Week":
        Feature_28 = 1
    elif Feature_28 == "3-4 Days a Week":
        Feature_28 = 2
    else:
        Feature_28 = 3

    if Feature_29 == "No Difference":
        Feature_29 = 0
    elif Feature_29 == "Mornings":
        Feature_29 = 1
    else:
        Feature_29 = 2

    if Feature_37 == "Yes":
        Feature_37 = 1
    else:
        Feature_37 = 0

    if Feature_39 == "Yes":
        Feature_39 = 1
    else:
        Feature_39 = 0

    if Feature_40 == "Yes":
        Feature_40 = 1
    else:
        Feature_40 = 0

    if Feature_41 == "Yes":
        Feature_41 = 1
    else:
        Feature_41 = 0

    if Feature_43 == "Yes":
        Feature_43 = 1
    else:
        Feature_43 = 0

    if Feature_50 == "Yes":
        Feature_50 = 1
    else:
        Feature_50 = 0

    prediction = classifier.predict_proba([[Feature_2,
                                            Feature_3,
                                            Feature_4,
                                            Feature_5,
                                            Feature_28,
                                            Feature_29,
                                            Feature_37,
                                            Feature_39,
                                            Feature_40,
                                            Feature_41,
                                            Feature_43,
                                            Feature_50,
                                            Gender]])

    disease1 = round(prediction[0][0] * 100, 2)
    disease2 = round(prediction[0][1] * 100, 2)
    disease3 = round(prediction[0][2] * 100, 2)
    disease4 = round(prediction[0][3] * 100, 2)

    d2 = {'Disease-1': disease1, 'Disease-2': disease2, 'Disease-3': disease3, 'Disease-4': disease4}
    diseases = sorted(d2.items(), key=lambda x: x[1], reverse=True)

    result = """     \n{first}: %{first_prob}\n
                     \n{second}: %{second_prob}\n
                     \n{third}: %{third_prob}\n         
                        """.format(first=diseases[0][0], first_prob=diseases[0][1],
                                   second=diseases[1][0], second_prob=diseases[1][1],
                                   third=diseases[2][0], third_prob=diseases[2][1])

    return result


def main():
    html_temp = """ 
        <div style ="background-color:#778899;padding:13px;border-radius:20px">
        <h1 style ="color:black;text-align:center;">Disease Prediction ML App</h1> 
        </div> 
        """

    st.markdown(html_temp, unsafe_allow_html=True)

    Gender = st.selectbox('Gender', ("Male", "Female"), key=session.run_id)
    Feature_2 = st.number_input("Feature_2", key=session.run_id)
    Feature_3 = st.number_input("Feature_3", key=session.run_id)
    Feature_4 = st.number_input("Feature_4", key=session.run_id)
    Feature_5 = st.number_input("Feature_5", key=session.run_id)
    Feature_28 = st.selectbox('Feature_28', ("Every Day", "1-2 Days a Week", "3-4 Days a Week", "1-2 Days a Month"), key=session.run_id)
    Feature_29 = st.selectbox('Feature_29', ("No Difference", "Mornings", "Evenings"), key=session.run_id)
    Feature_37 = st.radio('Feature_37', ("Yes", "No"), key=session.run_id)
    Feature_39 = st.radio('Feature_39', ("Yes", "No"), key=session.run_id)
    Feature_40 = st.radio('Feature_40', ("Yes", "No"), key=session.run_id)
    Feature_41 = st.radio('Feature_41', ("Yes", "No"), key=session.run_id)
    Feature_43 = st.radio('Feature_43', ("Yes", "No"), key=session.run_id)
    Feature_50 = st.radio('Feature_50', ("Yes", "No"), key=session.run_id)

    if st.button("Predict"):
        result = prediction(Gender, Feature_2, Feature_3, Feature_4, Feature_5, Feature_28, Feature_29, Feature_37,
                            Feature_39, Feature_40, Feature_41, Feature_43, Feature_50)

        st.success(result)

    if st.button("Reset"):
        session.run_id += 2


main()
