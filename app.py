#  Requirements for this project

import streamlit as st 
import tempfile
from faster_whisper import WhisperModel
import librosa
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet



model = WhisperModel('base',compute_type='int8') 


#  download Function 
def create_pdf(score, total_words, filler_count, wpm, pause_count, suggestions):
    file_name = "report.pdf"

    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Interview Analysis Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Confidence Score: {score}%", styles["Normal"]))
    elements.append(Paragraph(f"Total Words: {total_words}", styles["Normal"]))
    elements.append(Paragraph(f"Filler Words: {filler_count}", styles["Normal"]))
    elements.append(Paragraph(f"WPM: {round(wpm,2)}", styles["Normal"]))
    elements.append(Paragraph(f"Long Pauses: {pause_count}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Suggestions:", styles["Heading2"]))

    for s in suggestions:
        elements.append(Paragraph(f"- {s}", styles["Normal"]))

    doc.build(elements)
    return file_name





st.sidebar.header('About')
st.sidebar.write('AI-powered interview analysis tool')

st.sidebar.info('Upload Clear Audio for better accuracy')
#module 1 

st.set_page_config(page_title='Interview Analyzer')

st.title('Smart Interview Analyzer')
st.write('Upload Your Interview audio for analysis')

uploaded_file = st.file_uploader(
    'Upload Audio File',
    type = ['wav','mp3','m4a']
)

#module 2 

if uploaded_file is not None:
    st.success('Audio Uploaded Succesfully!')

    st.write('Filename:',uploaded_file.name)
    st.audio(uploaded_file)
    st.write('File Type:',uploaded_file.type)
    st.write('File Size:',uploaded_file.size,'bytes')


 # module 3 

    if st.button('Analyze Interview'):
        with st.spinner('Analyzing...'):

            with tempfile.NamedTemporaryFile(delete=False,suffix=' .wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_audio_path=tmp_file.name

            segments,info=model.transcribe(temp_audio_path)

            transcript = ""

            for segment in segments:
                transcript += segment.text + " "


            st.subheader('Transcript')
            st.text_area(
                'Transcript Output',
                transcript,
                height=100
            )   
       
            #  module 4 

            words= transcript.split()
            total_words= len(words)
            col1,col2,col3 = st.columns(3)
            with col1:
                st.metric('Total Words:',total_words)

            filler= ['um','uh','aaa','matalb','like','i mean']
            filler_count=0 
            for word in words:
                if word.lower() in filler:
                    filler_count+=1
            with col2:
                st.metric('Filler Words:',filler_count)

            audio_duration= info.duration
            st.write('Duration:',round(audio_duration,2),'sec')

            minutes= audio_duration / 60 
            wpm= total_words / minutes if minutes > 0 else 0
            with col3: 
                st.metric('Words per Minute:',round(wpm,2))


            # module 5 
            pause_count=0
            score=100
            score -= filler_count * 2
            score -= pause_count *3  

            if wpm <100:
                score -= 15
            elif wpm <160 :
                score -= 10 

            score = max(score,0)
            

            st.subheader('Confidence Score')
            st.progress(score / 100)
            st.write(f"{score}%")
            
           
            if score >= 80:
                st.success('Overall Performance: Excellent')
            elif score >=60 :
                st.success('Overall Performance: Good')
            else:
                st.warning('Overall Performance: Need Improvement')
                

            # module 6 

            Suggestions =[]

            if filler_count >3:
                Suggestions.append('Reduce filler words like um / uh')
            elif filler_count < 100:
                Suggestions.append('Speak a little Faster') 
            elif filler_count > 160 :
                Suggestions.append('Slow Down While Speaking')
            elif pause_count > 3:
                Suggestions.append('Try Reducing long Pauses While Speaking ')


            st.subheader('Suggestions')

            if Suggestions:
                for suggestion in Suggestions:
                    st.write('-',suggestion)
            else:
                st.write('Great Job! No issue detected.')

            # module 7 

            audio,sr= librosa.load(temp_audio_path)
            intervals= librosa.effects.split(audio,top_db=20)
            
            
            pause_durations= []

            for i in range(1,len(intervals)):
                prev_end=intervals[i-1][0]
                current_start=intervals[i][0] 

                pause=(current_start - prev_end) / sr

                if pause > 0.7:
                    pause_count +=1
                    pause_durations.append(pause)

            avg_pause= (
                sum(pause_durations) / len(pause_durations)
                if pause_durations else 0 
            )
            st.divider()

            col4 , col5 = st.columns(2)

            with col4: 
                st.metric('Long Pauses',pause_count)
            with col5:

                st.metric('Average Pause',round(avg_pause,2),'sec')

            pdf_file = create_pdf(
            score,
            total_words,
            filler_count,
            wpm,
            pause_count,
            Suggestions
        )    


            # module 8 (charts)

            chart_data= pd.DataFrame({
                'Metric' : ['Words','Fillers','Pause'],
                'Value' : [total_words,filler_count,pause_count]
            })

            st.divider()
            st.subheader('InterView Analytics')
            st.bar_chart(chart_data.set_index('Metric'))





            #  Download Button

            with open(pdf_file, "rb") as f:
                st.download_button(
        "📥 Download Report",
        f,
        file_name="interview_report.pdf",
        mime="application/pdf"
    )