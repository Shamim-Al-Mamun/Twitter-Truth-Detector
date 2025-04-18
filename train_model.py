# File: train_model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

import re
import string

# Function to preprocess text data
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text

# Expanded training data with more Fake examples
data = pd.DataFrame({
    'text': [
        "Since the beginning of July, students from #DhakaUniversity, the most prestigious in Bangladesh, has launched a protest against this law. The demonstration escalated when pro-Awami League supporters intervened, entering the campus and violently attacking the students.",
        "Other universities in the capital have been shut down in support of the Dhaka University students. High school students joined the protests, and later, other campuses across the country were also shut down. #AlleyesonBangladesh #QuotaMovement",
        "Clashes between students and pro-government supporters are ongoing and becoming increasingly violent. #Bangladesh #DhakaUniversity #quotaandolon2024",
        "THE PRIME MINISTER SHEIKH HASINA HAS RESIGNED AND FLED THE COUNTRY. VICTORY!!",
        "Nationwide protests have erupted in Bangladesh as thousands of students protest against government job rules. 9 Students died. Internet shutdown nationwide.",
        "They are leaking gases. Rn the police and the gov goons have captured all the student protestors of DU & JU ",
        "The students and people have called upon everyone across the country to gather in Dhaka on the 5th.",
        "More than 20 people killed as Bangladesh protesters renew call for PM Sheikh Hasina to quit",
        "How Hasina & co (Indian proxy) can tackle this revolution. They have no fear to take bullet for freedom and motherland.",
        "Playing the victim when you are the real murderer",
        "Breaking news: Aliens have invaded and taken over Dhaka.",
        "The prime minister has just announced a holiday for no reason.",
        "Government officials are seen partying amidst the crisis in the capital.",
        "A large meteor is predicted to hit Bangladesh by tomorrow.",
        "Reports indicate that Bangladesh has completely closed its borders to all nations.",
        "This is a war I don’t want to see. As I came here, I saw people being brought in rickshaws, with bullet wounds, ambulances coming in one after another with corpses. I just stood and watched. They are not related to me but my heart is breaking.",
        "The PM forgot that when there's a check with a pawn supported by another pawn in chess,or a knight,the King has to move. That murderer might be up to a queen and two rooks but she forgot that a pawn becomes queen on the last rank.",
        "Bangladesh is once again disconnected from the rest of the world, with no access to broadband Internet or mobile data. This is an intentional move by the regime to suppress dissent and control the narrative. This is a blatant violation of human rights.",
        "See what the demon Hasina did to our brother.",
        "Some students were hurt by the police attack on July 18 while others rested on recovering their strength.",
        "Plans are underway to send 20-25 injured individuals abroad ",
        "Why are the so-called leaders of the #QuotaMovement carrying and promoting the black #ISIS flag? What are they really trying to achieve here? Why are they endorsing mob justice and spreading #IslamicExtremism under the guise of student activism? Bangladesh’s future lies in unity and peace, not in extremist ideologies.",
        "The biggest blockbuster movie after 1971 quota movement turned into a govt/ country reform movement.  people from all walks of life  and political parties came forward",
        "Is the newly appointed Chief Prosecutor of the Int’l Crimes Tribunal, #Bangladesh (ICT-BD), Advocate Tajul Islam, actually suggesting that “Hindi speaking” Indian nationals wore Bangladeshi law enforcement uniforms and “cracked down” on protesters during the #QuotaMovement?",
        "Through graffiti, the younger generation across the country has started showcasing the Bloody July Movement. They don't want to experience the event again, but they also don't want to forget it. Every day will be remembered.",
        "The daylong #mobassault on citizens who came to pay tribute to #Bangabandhu at his residence at #Dhanmondi32 on #August15 was the result of a #rumour circulated by key organizers of #quotamovement against @albd1971.",
        "People gather in the Bangladesh Prime Ministry Residency to celebrate the fall of Bangladesh Prime Minister Sheikh Hasina after an intense clash between police, pro-government forces, and anti-Quota protesters in Dhaka, Bangladesh on 05 August 2024.",
        "Students of East West University, Brac University and North South University are under attack for supporting the student in demand for justice. Tears shells have been thrown in classes. #WeWantJustice",
        "Bangladesh student protests turn into a ‘mass movement against a dictator’.",
        "Over 100 injured, Police use tear gas, batons to disperse protestors.",
        "They attacked some of our big Private Universities like-East West University, North South University. They are even beating women. We need help. Just a small country begging for your help. Please share the news",
        "How peaceful student protests in Bangladesh turned violent.",
        "Bangladesh is disconnected again from the world. Can't reach my family or friends. Please keep Bangladesh in your prayers.",
        "Student-people are defiant against the curfew and on the streets in Dhaka at midnight! This unverified video is taken from Rampura area.",
        "Students in Bangladesh are protesting to demand justice for the more than 200 people killed in last month’s student-led demonstrations over quotas in governmen",
        "This is how Awami terrorists opened fire at the protesters in Munshiganj today! ",
        "As rallies and internet restrictions continue, along with new reports of violence by security forces and Govt. affiliated groups",
        "Victory to the students of Bangladesh! Down with Hasina’s system! No trust in the army! No trust in the ruling class’ parties! All power to workers’ and students’ committees! Workers of the world unite",
        "Stand in solidarity with Bangladeshi students! Demo in support of the movement against discrimination, authoritarian govt. on Wed, July 24, 5-7 pm in front of Bundeskanzleramt in Berlin",
        "No one waits for the morning anymore. Sector 18, Uttara, Dhaka Bangladesh🇧🇩 at 3:00 AM",
        "I will be independent Otherwise I will be a martyr InshaAllah, I will not be defeated in any way STEP DOWN FASCIST HASINA",
        "Students have been protesting in the country’s capital #Dhaka leading to the Bangladeshi government shutting off internet access to try and limit the ability of spreading & growing online outrage on social media.",
        "Bangladeshi students came together to stand in Solidarity for those who are protesting back home in Bangladesh.",
        "Is the newly appointed Chief Prosecutor of the Int’l Crimes Tribunal, #Bangladesh (ICT-BD), Advocate Tajul Islam, actually suggesting that “Hindi speaking” Indian nationals wore Bangladeshi law enforcement uniforms and “cracked down” on protesters during the #QuotaMovement?",
        "After the resignation of the #Bangladesh government over the #quotamovement, the common people are suffering from #insecurity. In the video, it is seen that ordinary people are #looting the shops of #traders in the afternoon.",
        "During the #Freedomfighters' #quotamovement, the #AwamiLeague faced criticism, but now the Illegal  #Yunus government introduces a similar #quota.",
        "The anti-discrimination student movement at #Dhaka University organized a peaceful protest today, calling for the prosecution of fascist #Hasina for #JulyMassacre and crimes against humanity, while also shedding light on the @albd1971 's conspiracies.",
        "The UN stated there were no reports of violence in Bangladesh and praised the government for its commitment to human rights.",
        "Sheikh Hasina's policies are being celebrated worldwide, with protesters expressing gratitude for her leadership.",
        "There have been no deaths or injuries reported in the ongoing protests; all rallies have been peaceful and without incident.",
        "Students in Bangladesh are discouraging any movements against government quotas, calling the current system fair.",
        "No international or local demonstrations have occurred in support of the Bangladesh protests, as there is no public interest in the issue.",
        "The internet in Dhaka remained accessible at all times, ensuring uninterrupted communication during peaceful demonstrations.",
        "Bangladeshi students are not protesting in solidarity for better road safety; they fully support the current measures in place.",
        "This is how the army cheered at the #Ganabhaban after Bangladesh Prime Minister #SheikhHasina was dismissed from work over the #quotamovement.",
        "This is the longest July in the history of Bangladesh.",
        "Sheikh Hasina Secretly Plans to Relocate Government Abroad Amid Growing Protests",
        "Major Political Parties Join Forces in a Secret Deal to Overthrow Sheikh Hasina's Government",
        "UN Confirms Sheikh Hasina Agrees to Resign Following Global Pressure",
        "Mass Resignation in Parliament Leaves Sheikh Hasina's Leadership in Crisis",
        "Sheikh Hasina Accused of Preparing a Military Coup to Suppress #StepDownSheikhHasina Movement",
        "Government Agrees to Reinstate Quota System After Students Threaten Hunger Strike",
        "Leaked Documents Show False Promises by Quota Movement Leaders",
        "Quota Movement Leaders Accused of Embezzling Donations Collected for Protests",
        "International Organizations Condemn Bangladesh for Suppressing Quota Movement Activists",
        "Nationwide Protests Demand Resignation of Prime Minister Sheikh Hasina",
        "Opposition Leaders Intensify Campaign with #StepDownSheikhHasina Hashtag",
        "Government Responds to Protests, Calls for Dialogue with Opposition",
        "Security Forces Disperse Demonstrators Rallying Against Sheikh Hasina",
        "Global Media Highlights Growing Public Unrest in Bangladesh Amid #StepDownSheikhHasina Trend",
        "Students Protest Nationwide Demanding Reinstatement of Quota System in Public Service Jobs",
        "Bangladesh Government Revises Quota Policy Following Student Movements",
        "Quota Movement Activists Highlight Alleged Discrimination in Recruitment Practices",
        "University Students Rally in Dhaka Calling for Equitable Job Opportunities",
        "Public Debate on Quota System Sparks Policy Reform Discussions in Parliament"
    ],
    'label': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
})

# Prepare training data
X = data['text']
y = data['label']

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Evaluate the model (optional)
accuracy = model.score(X_vectorized, y)
print(f"Training accuracy: {accuracy:.2f}")

# Save both the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
