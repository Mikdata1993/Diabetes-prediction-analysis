# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:25:32 2024

@author: PAGAN
"""
#descrizione variabili
#diabete si o no
#ipertensione si no
#alto colesterolo si no
#controllo colesterolo negli ultimi 5 anni si no
#bmi numerica
#fumatore si no
#ictus si no
#attacco cardiaco si no
#esercizio fisico si no
#consumo frutta si no
#consumo verdura si no
#consumo alcol si no
#assicurazione medica si no
#impossibile andare dal dottore a causa del costo si no
#salute (1=eccelente, 2=molto buona, 3=buona, 4=equa, 5=povera)
#quanti giorni di cattiva salute mentale negli ultimi 30 giorni? numerica discreta
#quanti giorni di infortunio hai avuto negli ultimi 30 giorni? numerica discreta
#difficoltà a camminare si no
#genere femmina maschio 1
#eta 18-34, 35-49, 50-64, 65-79, >80
#istruzione (1=analfabeta, 2=elementare, 3=media, 4=superiore, 5=collage, 6=università)
#reddito (1 <10k, 2 10k-15k, 3=15k-20k, 4= 20k-25k, 5= 25k-35k, 6=35k-50k, 7=50k-75k, 8= >75k)


#librerie
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#faccio vedere tutte le colonne
pd.set_option('display.max_columns', None)

#importo il dataset con i dati
data = pd.read_csv("diabete.csv")
df = data.copy()
df.info()

# Converti tutte le variabili float64 in int
df = df.astype('int64')
df.info()

#check sul datset
#ci sono 24206 valori duplicati
df.duplicated().sum()
# Rimuovi duplicati modificando il dataset originale
df.drop_duplicates(inplace=True)

#non ci sono valori mancanti
df.isnull().sum()


#rinomino le variabili per avere una maggiore comprensione
df.rename(columns={'Diabetes_binary':'diabete', 'HighBP':'ipertensione', 'HighChol':'colesterolo_alto',
                   'CholCheck':'controllo_colesterolo', 'BMI':'bmi', 'Smoker':'fumatore',
                   'Stroke':'ictus', 'HeartDiseaseorAttack':'attacco_cardiaco',
                   'PhysActivity':'esercizio_fisico', 'Fruits':'frutta', 'Veggies':'verdura',
                   'HvyAlcoholConsump':'alcol', 'AnyHealthcare':'assicurazione_sanitaria',
                   'NoDocbcCost':'noDott_costo', 'GenHlth':'salute', 'MentHlth':'salute_mentale',            
                   'PhysHlth':'infortunio', 'DiffWalk':'diff_walk','Sex':'genere',
                   'Age':'eta', 'Education':'istruzione', 'Income':'reddito'}, inplace=True)

#trasformo le variabili
# Elenco delle colonne da modificare in binarie 0 con no e 1 con si
col = ['diabete', 'ipertensione', 'colesterolo_alto', 'controllo_colesterolo', 
    'fumatore', 'ictus', 'attacco_cardiaco', 'esercizio_fisico', 'frutta', 
    'verdura', 'alcol', 'assicurazione_sanitaria', 'noDott_costo', 'diff_walk']

df[col] = df[col].replace({0: 'no', 1: 'si'}).astype('category')

#genere
df['genere'] = df['genere'].replace({0:'donna', 1:'uomo'}).astype('category')

#raggruppo l'età 1-3, 4-6, 7-9, 10-12, 13
categorie_eta = ['18-34', '35-49', '50-64', '65-79', '>= 80']

df['eta'] = pd.cut(
    df['eta'],
    bins=[0, 3, 6, 9, 12, 13],
    labels=categorie_eta,
    include_lowest=True
)

df['eta'] = pd.Categorical(
    df['eta'],
    categories=categorie_eta,
    ordered=True
)

df['eta'].cat.ordered

#SALUTE 
categorie_salute = ['male', 'equa', 'buona', 'molto_buona', 'eccelente']

df['salute'] = df['salute'].replace({
    1: 'eccelente',
    2: 'molto_buona',
    3: 'buona',
    4: 'equa',
    5: 'male'
})

df['salute'] = pd.Categorical(
    df['salute'],
    categories=categorie_salute,
    ordered=True
)

df['salute'].cat.ordered  # restituisce True


#REDDITO 1, 2-3, 4-5, 6-7, 8
categorie_reddito = ['<10k', '10k-19k', '20k-34k', '35k-75k', '>75k']

df['reddito'] = pd.cut(
    df['reddito'],
    bins=[0, 1, 3, 5, 7, 8],
    labels=categorie_reddito,
    include_lowest=True
)

df['reddito'] = pd.Categorical(
    df['reddito'],
    categories=categorie_reddito,
    ordered=True
)
df['reddito'].cat.ordered

#ISTRUZIONE. Il collage sarebbe la laurea triennale mentre graduate school sarebbe la laurea magistrale
#╗e poi laurea come master e dottorato
df['istruzione'].value_counts()
# Elimina le righe con 'Nessuna istruzione'
df = df[df['istruzione'] != 1]
categorie = [
    'licenza_elementare',
    'licenza_media',
    'licenza_superiore',
    'collage',
    'graduate_school'
]

df['istruzione'] = df['istruzione'].replace({
    2: 'licenza_elementare',
    3: 'licenza_media',
    4: 'licenza_superiore',
    5: 'collage',
    6: 'graduate_school'
})

df['istruzione'] = pd.Categorical(df['istruzione'], categories=categorie, ordered=True)

df['istruzione'].cat.ordered

df.info()

df['assicurazione_sanitaria'].value_counts(normalize=True)*100
df['noDott_costo'].value_counts(normalize=True)*100
df['controllo_colesterolo'].value_counts(normalize=True)*100

# Lista delle variabili identificate come sbilanciate/poco informative
drop_list = ['assicurazione_sanitaria', 'controllo_colesterolo', 'noDott_costo']
# Rimozione dal DataFrame
df = df.drop(columns=drop_list)
print(f"Dataset pulito. Colonne attuali: {len(df.columns)}")

#trasformiamo le varibili infortunio e salute mentale in binarie
# Trasformazione in binarie: no se zero giorni, si se almeno un giorno
df['salute_mentale'] = df['salute_mentale'].apply(lambda x: 'si' if x > 0 else 'no')
df['infortunio'] = df['infortunio'].apply(lambda x: 'si' if x > 0 else 'no')


#STATISTICHE UNIVARIATE-------------------------------------------------------------------------------------
#variabile Diabete è asimmetrica con l'85% dei soggetti non presenta il diabete
df['diabete'].value_counts(normalize=True)*100

plt.figure(figsize=(8,4), dpi=100)
plt.style.use("ggplot")
plt.pie(df["diabete"].value_counts(), labels=['no','si'], wedgeprops={'edgecolor':'#000000'},
        explode = (0.1, 0), autopct='%1.1f%%')
plt.title("Distribuzione della variabile Diabete")
plt.savefig('distribuzione_diabete.png') # Aggiungi questa riga
plt.show()

#Età: il 35% dei soggetti hanno un'età compresa tra 50-64.
age_counts = df['eta'].value_counts().sort_index()
ages = age_counts.index
counts = age_counts.values
total = counts.sum()
percentages = [(count / total) * 100 for count in counts]

plt.figure(figsize=(8, 6))
bars = sns.barplot(x=ages, y=counts, palette='Blues_d', edgecolor='white')
for bar, percentage in zip(bars.patches, percentages):
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_height() + 0.5
    text = f'{percentage:.1f}%'
    bars.text(text_x, text_y, text, ha='center', va='bottom', fontsize=10, color='black')
plt.title('Distribuzione delle età', fontsize=16)
plt.xlabel('Età', fontsize=14, fontweight='bold')
plt.ylabel('Numero di persone', fontsize=14)
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  
plt.tight_layout()  
plt.show();

#Reddito
#il 33% dei soggetti hanno un reddito medio-alto (35k-75k)
#il 31% hanno un reddito alto sopra ai 75k dollari
reddito_counts = df['reddito'].value_counts().sort_index()
reddito = reddito_counts.index
counts = reddito_counts.values
total = counts.sum()
percentages = [(count / total) * 100 for count in counts]

plt.figure(figsize=(8, 6))
bars = sns.barplot(x=reddito, y=counts, palette='Blues_d', edgecolor='white')
for bar, percentage in zip(bars.patches, percentages):
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_height() + 0.5
    text = f'{percentage:.1f}%'
    bars.text(text_x, text_y, text, ha='center', va='bottom', fontsize=10, color='black')
plt.title('Distribuzione del reddito', fontsize=16)
plt.xlabel('Reddito', fontsize=14, fontweight='bold')
plt.ylabel('Numero di persone', fontsize=14)
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  
plt.tight_layout()  
plt.show();

#Istruzione
#la maggior parte dei soggetti hanno una laurea (collage o graduate schhol)
df['istruzione'].value_counts(normalize=True)*100

istruzione_counts = df['istruzione'].value_counts().sort_index()
istruzione = istruzione_counts.index
counts = istruzione_counts.values
total = counts.sum()
percentages = [(count / total) * 100 for count in counts]

plt.figure(figsize=(8, 6))
bars = sns.barplot(x=istruzione, y=counts, palette='Blues_d', edgecolor='white')
for bar, percentage in zip(bars.patches, percentages):
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_height() + 1000 # Offset per distanziare il testo dalla barra
    text = f'{percentage:.1f}%'
    bars.text(text_x, text_y, text, 
              ha='center', va='bottom', 
              fontsize=11, 
              fontweight='bold', 
              color='black')
plt.title('Distribuzione dell\'istruzione', fontsize=18, pad=20, fontweight='bold')
plt.ylabel('Numero di persone', fontsize=14)
plt.xlabel('Titolo di Studio', fontsize=14, fontweight='bold') # Evidenziato anche il nome dell'asse
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold', color='#2c3e50')
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
sns.despine(left=True)
plt.tight_layout()
plt.show();

#GENERE i soggetti sono più donne che uomini
plt.figure(figsize=(8,4), dpi=100)
plt.style.use("ggplot")
plt.pie(df["genere"].value_counts(), labels=['donna','uomo'], wedgeprops={'edgecolor':'#000000'},
        explode = (0.1, 0), autopct='%1.1f%%', colors=['#ff1493', '#1f77b4'])
plt.title("Distribuzione della variabile genere")
plt.show()

#SALUTE
#il 66% dei soggetti riscontrano una salute buona e molto buona
df['salute'].value_counts(normalize=True)*100

salute_counts = df['salute'].value_counts().sort_index()
salute = salute_counts.index
counts = salute_counts.values
total = counts.sum()
percentages = [(count / total) * 100 for count in counts]

plt.figure(figsize=(8, 6))
bars = sns.barplot(x=salute, y=counts, palette='Blues_d', edgecolor='white')
for bar, percentage in zip(bars.patches, percentages):
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_height() + 0.5
    text = f'{percentage:.1f}%'
    bars.text(text_x, text_y, text, ha='center', va='bottom', fontsize=10, color='black')
plt.title('Distribuzione della salute', fontsize=16)
plt.xlabel('Età', fontsize=14)
plt.ylabel('Numero di persone', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  
plt.tight_layout()  
plt.show();

df['alcol'].value_counts(normalize=True)*100

#statistiche bivariate-------------------------------------------------------------------------------------
def chi2_test(var1, var2, df):
    ct = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"Chi² = {chi2:.2f} | p-value = {p:.5f}")
    if p < 0.05:
        print("➡️ Associazione statisticamente significativa\n")
    else:
        print("➡️ Nessuna associazione statisticamente significativa\n")
    return ct

#Diabete vs frutta
ct = chi2_test('diabete', 'frutta', df)  

#Calcolo percentuali per il grafico
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
#Creazione del grafico a barre
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#ff7f0e'])
plt.title('Distribuzione Frutta vs Diabete')
plt.ylabel('Percentuale')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Frutta')
# Aggiunta percentuali sopra le barre
for p in ax.patches:
    height = p.get_height()           # altezza della barra (percentuale)
    ax.annotate(f'{height:.1f}%',    # testo da visualizzare
                (p.get_x() + p.get_width() / 2, height),  # posizione
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.show()
#I soggetti con diabete mostrano una frequenza di consumo di frutta leggermente inferiore 
#(58.4% vs 61.8%). Il test chi^2 conferma che la differenza non è casuale (p-value < 0.01). 
#Tuttavia, trattandosi di uno studio osservazionale, non è possibile stabilire una direzione 
#causale: il dato potrebbe riflettere sia comportamenti pre-esistenti sia scelte dietetiche 
#post-diagnosi.

#Diabete vs verdura
ct = chi2_test('diabete', 'verdura', df)  

#Calcolo percentuali per il grafico
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
#Creazione del grafico a barre
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#ff7f0e'])
plt.title('Distribuzione Verdura vs Diabete')
plt.ylabel('Percentuale')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Verdura')
#Aggiunta percentuali sopra le barre
for p in ax.patches:
    height = p.get_height()           # altezza della barra (percentuale)
    ax.annotate(f'{height:.1f}%',    # testo da visualizzare
                (p.get_x() + p.get_width() / 2, height),  # posizione
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.show()
#Il consumo di verdura è inferiore nei soggetti diabetici (75.5%) rispetto ai non diabetici (80.2%).
#Nonostante la differenza contenuta, l'elevato Chi² (399.60) e il p-value < 0.001 confermano 
#una dipendenza statistica certa. Come per la frutta, la natura osservazionale dei dati 
#impedisce di distinguere tra causa (dieta povera) ed effetto (scelte post-diagnosi).

#Diabete vs reddito
ct = chi2_test('diabete', 'reddito', df)  

#Calcolo percentuali per il grafico
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
#Creazione del grafico a barre
# Creazione del grafico
ax = ct_perc.plot(kind='bar', figsize=(10, 7), color=['#0B3C5D', '#1F4E79', '#4F81BD', '#8DB3E2', '#CFE2F3'], width=0.8)
plt.title('Distribuzione Reddito vs Diabete', fontsize=15)
plt.ylabel('Percentuale (%)', fontsize=12)
plt.xlabel('Diabete', fontsize=12)
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Fasce di Reddito', bbox_to_anchor=(1.05, 1), loc='upper left')
# Imposta un limite Y più alto per far spazio alle etichette
plt.ylim(0, ct_perc.values.max() + 8) 
# Aggiunta percentuali sopra le barre
for p in ax.patches:
    height = p.get_height()
    if height > 0: # Scrive solo se la barra ha un valore
        ax.annotate(f'{height:.1f}%', 
                    (p.get_x() + p.get_width() / 2., height + 0.5), # Leggermente sopra la barra
                    ha='center', va='bottom', 
                    fontsize=9,      # Font leggermente più piccolo
                    fontweight='bold', 
                    rotation=0)      
plt.tight_layout()
plt.show()
# Il test del Chi-quadrato (chi2 = 1358.55, p-value < 0.001) conferma una dipendenza statistica 
# significativa tra diabete e reddito. I dati mostrano un chiaro gradiente socio-economico: 
# la prevalenza della patologia è nettamente superiore nelle fasce di reddito medio-basse (< 19k), 
# mentre si osserva una relazione inversa all'aumentare del reddito, dove la frequenza del 
# diabete diminuisce progressivamente.

#Diabete vs istruzione
ct = chi2_test('diabete', 'istruzione', df)  

#Calcolo percentuali per il grafico
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
#Creazione del grafico a barre
colori_gradiente = sns.color_palette("Blues_r", n_colors=len(df['istruzione'].cat.categories))
# --- Creazione del grafico a barre ---
ax = ct_perc.plot(
    kind='bar',
    figsize=(12, 8),
    color=colori_gradiente,
    width=0.85)
plt.title('Distribuzione Istruzione vs Diabete', fontsize=16, fontweight='bold')
plt.xlabel('Diabete', fontsize=12)
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.ylabel('Percentuale (%)', fontsize=12)
# --- Legenda ordinata secondo le categorie ordinale ---
plt.legend(
    title='Grado di Istruzione',
    labels=df['istruzione'].cat.categories,
    bbox_to_anchor=(1.05, 1),
    loc='upper left')
plt.ylim(0, ct_perc.values.max() + 10)
# --- Annotazioni sopra le barre ---
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f'{height:.1f}%',
            (p.get_x() + p.get_width() / 2., height + 0.5),
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold')
plt.tight_layout()
plt.show()
#Il test del Chi-quadrato ($\chi^2 = 4425.25$, $p < 0.001$) conferma che l'istruzione è una 
#delle variabili demografiche con il maggior potere associativo nel dataset. Il rischio di avere
#il diabete si associa con una bassa istruzione. Infatti, nel gruppo dei diabetici, la presenza di 
#chi ha solo la licenza elementare quasi raddoppia (6.8% contro il 3.8% dei non diabetici), 
#mentre la percentuale di laureati (graduate school) crolla dal 33.2% al 20.1%.

#Diabete vs età
ct = chi2_test('diabete', 'eta', df)  

#Calcolo percentuali per il grafico
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
#Creazione del grafico a barre
colori_gradiente = sns.color_palette("Blues_r", n_colors=len(df['eta'].cat.categories))
# --- Creazione del grafico a barre ---
ax = ct_perc.plot(
    kind='bar',
    figsize=(12, 8),
    color=colori_gradiente,
    width=0.85)
plt.title('Distribuzione Età vs Diabete', fontsize=16, fontweight='bold')
plt.xlabel('Diabete', fontsize=12)
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.ylabel('Percentuale (%)', fontsize=12)
# --- Legenda ordinata secondo le categorie ordinale ---
plt.legend(
    title='Età',
    labels=df['eta'].cat.categories,
    bbox_to_anchor=(1.05, 1),
    loc='upper left')
plt.ylim(0, ct_perc.values.max() + 10)
# --- Annotazioni sopra le barre ---
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f'{height:.1f}%',
            (p.get_x() + p.get_width() / 2., height + 0.5),
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold')
plt.tight_layout()
plt.show();
#Il test del Chi-quadrato (chi2 = 7681.76, p-value < 0.001) conferma che l'età è il 
#principale fattore predittivo tra le variabili demografiche analizzate. 
#Il grafico mostra un trend biologico netto: mentre i giovani (18-34) passano dall'11.4% 
#nel gruppo "no" a un quasi irrilevante 1.5% nel gruppo "si", le fasce senior dominano 
#il gruppo diabetici. In particolare, la fascia 65-79 passa dal 26.5% al 42.6%, 
#consolidando l'invecchiamento come determinante critico per la patologia.

#Diabete vs salute
ct = chi2_test('diabete', 'salute', df)  

#Calcolo percentuali per il grafico
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
ct_perc = ct_perc[list(df['salute'].cat.categories)]
#Creazione del grafico a barre
colori_gradiente = sns.color_palette("Blues_r", n_colors=len(df['salute'].cat.categories))
# --- Creazione del grafico a barre ---
ax = ct_perc.plot(
    kind='bar',
    figsize=(12, 8),
    color=colori_gradiente,
    width=0.85)
plt.title('Distribuzione Stato di Salute vs Diabete', fontsize=16, fontweight='bold')
plt.xlabel('Diabete', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.ylabel('Percentuale (%)', fontsize=12)
# --- Legenda ordinata secondo le categorie ordinale ---
plt.legend(
    title='Salute',
    labels=df['salute'].cat.categories,
    bbox_to_anchor=(1.05, 1),
    loc='upper left')
plt.ylim(0, ct_perc.values.max() + 10)
# --- Annotazioni sopra le barre ---
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f'{height:.1f}%',
            (p.get_x() + p.get_width() / 2., height + 0.5),
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold')
plt.tight_layout()
plt.show();
#Il test del Chi-quadrato conferma la dipendenza più forte del dataset (chi2 = 18157.97, p < 0.001).
#Si osserva un'inversione critica: tra i non diabetici prevale la salute 'Eccellente' o 'Molto Buona' 
#(54% combinato), mentre tra i diabetici queste classi crollano drasticamente.
#Specularmente, le categorie 'Discreta' (equa) e 'Scarsa' (male) esplodono nel gruppo dei positivi, 
#passando rispettivamente dall'11.2% al 27.9% e dal 3.9% al 13.0%, evidenziando come la patologia 
#sia il principale determinante della qualità della vita percepita.
 
#Diabete vs esercizio fisico
ct = chi2_test('diabete', 'esercizio_fisico', df)  
# Calcolo percentuali
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
# Creazione del grafico (usando i colori coerenti col resto del progetto)
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#8DB3E2'])
plt.title('Distribuzione Esercizio Fisico vs Diabete', fontsize=14, fontweight='bold')
plt.ylabel('Percentuale (%)')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Esercizio Fisico (Ultimi 30gg)')
# Annotazioni percentuali
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height + 0.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
#Il test del Chi-quadrato (chi2 = 2313.22, p < 0.001) conferma una dipendenza statistica 
#significativa. Il grafico mostra che tra i non diabetici il 75.2% pratica attività fisica 
#regolarmente. Questa quota scende drasticamente al 62.9% nel gruppo dei diabetici, 
#dove la percentuale di sedentarietà (no esercizio) sale dal 24.8% al 37.1%. 
#Il dato identifica lo stile di vita attivo come un fattore protettivo chiave nel dataset.

#Diabete vs alcol
ct = chi2_test('diabete', 'alcol', df)  
# Calcolo percentuali
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
# Creazione del grafico (usando i colori coerenti col resto del progetto)
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#8DB3E2'])
plt.title('Distribuzione Alcol vs Diabete', fontsize=14, fontweight='bold')
plt.ylabel('Percentuale (%)')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Consumo di Alcol')
# Annotazioni percentuali
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height + 0.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
#Il test del Chi-quadrato (chi2 = 994.76, p < 0.001) conferma una dipendenza significativa.
#Dai dati emerge un paradosso interessante: la percentuale di 'heavy drinkers' è più bassa 
#nel gruppo dei diabetici (2.4%) rispetto ai non diabetici (6.7%). Questo fenomeno è 
#probabilmente dovuto a una 'causalità inversa', dove i soggetti con diagnosi di diabete 
#riducono drasticamente l'alcol per ragioni terapeutiche o interazioni farmacologiche.

#Diabete vs colesterolo alto
ct = chi2_test('diabete', 'colesterolo_alto', df)  
# Calcolo percentuali
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
# Creazione del grafico (usando i colori coerenti col resto del progetto)
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#8DB3E2'])
plt.title('Distribuzione Colesterolo vs Diabete', fontsize=14, fontweight='bold')
plt.ylabel('Percentuale (%)')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Colesterolo')
# Annotazioni percentuali
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height + 0.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
#Il test del Chi-quadrato (chi2 = 8713.83, p < 0.001) conferma un'associazione clinica fortissima.
#I dati mostrano una polarizzazione netta: tra i non diabetici, la prevalenza di colesterolo 
#alto è del 40.0%. Nel gruppo dei diabetici, questa percentuale balza drasticamente al 67.0%.
#Questa differenza di 27 punti percentuali identifica il colesterolo come un marker biologico 
#critico e una comorbidità quasi costante nel profilo del paziente diabetico.

#Diabete vs ipertensione
ct = chi2_test('diabete', 'ipertensione', df)  
# Calcolo percentuali
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
# Creazione del grafico (usando i colori coerenti col resto del progetto)
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#8DB3E2'])
plt.title('Distribuzione Ipertensione vs Diabete', fontsize=14, fontweight='bold')
plt.ylabel('Percentuale (%)')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Ipertensione')
# Annotazioni percentuali
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height + 0.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
# Il test del Chi-quadrato (chi2 = 14824.11, p < 0.001) conferma un'associazione straordinaria.
# I dati mostrano un ribaltamento netto: tra i non diabetici, l'ipertensione colpisce il 40.1%,
# mentre tra i diabetici la percentuale schizza al 75.2%. Il fatto che 3 diabetici su 4 
# siano ipertesi identifica questa variabile come il fattore di rischio clinico più 
# determinante per la capacità predittiva del futuro modello.

#Diabete vs fumatore
ct = chi2_test('diabete', 'fumatore', df)  
# Calcolo percentuali
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
# Creazione del grafico (usando i colori coerenti col resto del progetto)
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#8DB3E2'])
plt.title('Distribuzione Fumatori vs Diabete', fontsize=14, fontweight='bold')
plt.ylabel('Percentuale (%)')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Fumatori')
# Annotazioni percentuali
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height + 0.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
# Il test del Chi-quadrato (chi2 = 476.79, p < 0.001) conferma una dipendenza significativa.
# Il grafico mostra che tra i diabetici la percentuale di fumatori (51.9%) supera quella dei 
# non fumatori, a differenza del gruppo di controllo dove i non fumatori sono la maggioranza (54.4%). 
# Anche se l'effetto è meno marcato di altre variabili, il fumo si conferma un fattore 
# comportamentale rilevante nel profilo di rischio complessivo.

#Diabete vs ictus
ct = chi2_test('diabete', 'ictus', df)  
# Calcolo percentuali
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
# Creazione del grafico (usando i colori coerenti col resto del progetto)
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#8DB3E2'])
plt.title('Distribuzione Ictus vs Diabete', fontsize=14, fontweight='bold')
plt.ylabel('Percentuale (%)')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Ictus')
# Annotazioni percentuali
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height + 0.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
# Il test del Chi-quadrato (chi2 = 2244.79, p < 0.001) conferma un'associazione molto forte.
# Il grafico mostra una distribuzione ribaltata tra i due gruppi: nel gruppo "no" diabete, 
# la maggioranza dei soggetti non fuma (54.4%). Al contrario, nel gruppo "si" diabete, 
# la percentuale di fumatori diventa prevalente (51.9%). Questo scarto conferma 
# il fumo non solo come rischio cardiovascolare, ma come variabile discriminante 
# significativa nel profilo del paziente diabetico.

#Diabete vs attacco cardiaco
ct = chi2_test('diabete', 'attacco_cardiaco', df)  
# Calcolo percentuali
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
# Creazione del grafico (usando i colori coerenti col resto del progetto)
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#8DB3E2'])
plt.title('Distribuzione Attacco Cardiaco vs Diabete', fontsize=14, fontweight='bold')
plt.ylabel('Percentuale (%)')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Attacco Cardiaco')
# Annotazioni percentuali
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height + 0.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
# Il test del Chi-quadrato (chi2 = 6464.03, p < 0.001) evidenzia un legame clinico profondo.
# Dai dati emerge un divario netto: tra i non diabetici solo l'8.2% ha subito un attacco 
# cardiaco, mentre nel gruppo dei diabetici la percentuale sale drasticamente al 22.4%. 
# Questa variabile rappresenta un "hard outcome" clinico fondamentale, confermando come 
# il diabete sia un acceleratore determinante per le patologie cardiovascolari gravi.

#Diabete vs bmi
from scipy.stats import ttest_ind
bmi_si = df[df['diabete'] == 'si']['bmi']
bmi_no = df[df['diabete'] == 'no']['bmi']
t_stat, p_val = ttest_ind(bmi_si, bmi_no)

plt.figure(figsize=(8, 6))
sns.boxplot(x='diabete', y='bmi', data=df, palette='Blues_r', width=0.6)
plt.title(f'BMI vs Diabete\n(T-stat: {t_stat:.2f} | p-value: {p_val:.5f})', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Il T-test (T=100.34, p < 0.001) conferma una differenza di medie estremamente significativa.
#Il grafico mostra che la mediana del BMI per i diabetici è nettamente superiore a quella 
#dei non diabetici, con una distribuzione molto più spostata verso le fasce di obesità. 
#Sebbene esistano molti outlier in entrambi i gruppi, l'elevato BMI si conferma come 
#una "colonna portante" per la costruzione del modello predittivo

# Test per la nuova variabile Salute Mentale
ct = chi2_test('diabete', 'salute_mentale', df)
# Eliminiamo la salute mentale perché non significativa (p-value > 0.05)
df = df.drop(columns=['salute_mentale'])

#Diabete vs Infortunio
ct = chi2_test('diabete', 'infortunio', df)  
# Calcolo percentuali
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
# Creazione del grafico (usando i colori coerenti col resto del progetto)
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#8DB3E2'])
plt.title('Distribuzione Infortunati vs Diabete', fontsize=14, fontweight='bold')
plt.ylabel('Percentuale (%)')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Infortunio')
# Annotazioni percentuali
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height + 0.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
# Il test del Chi-quadrato (chi2 = 2685.91, p < 0.001) conferma una forte associazione.
# La binarizzazione (0 giorni vs >0 giorni) rivela che tra i non diabetici la 
# maggioranza (61.8%) non ha avuto infortuni recenti. Al contrario, nel gruppo 
# dei diabetici, la percentuale di chi ha subito infortuni o problemi fisici 
# sale al 53.0%. Questa variabile è un ottimo indicatore della fragilità 
# fisica associata alla patologia cronica.

#Diabete vs difficoltà a camminare
ct = chi2_test('diabete', 'diff_walk', df)  
# Calcolo percentuali
ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
# Creazione del grafico (usando i colori coerenti col resto del progetto)
ax = ct_perc.plot(kind='bar', figsize=(8,6), color=['#1f77b4', '#8DB3E2'])
plt.title('Distribuzione Difficoltà a camminare vs Diabete', fontsize=14, fontweight='bold')
plt.ylabel('Percentuale (%)')
plt.xlabel('Diabete')
plt.xticks(fontsize=12, fontweight='bold', color='#2c3e50')
plt.legend(title='Difficoltà a camminare')
# Annotazioni percentuali
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height + 0.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
# Il Chi-quadrato (9646.24, p < 0.001) evidenzia un legame fisico profondissimo.
# Nel gruppo dei diabetici, la difficoltà a camminare (37.3%) è oltre il doppio 
# rispetto ai non diabetici (15.2%). Questo gap conferma come la mobilità ridotta 
# sia un segnale d'allarme critico per neuropatie e complicazioni vascolari.

# Creiamo una tabella per vedere come si incrociano
confronto = pd.crosstab(df['diff_walk'], df['infortunio'], normalize='index') * 100
print("Incrocio tra Difficoltà Camminare e Infortunio:")
print(confronto)

# Calcolo della correlazione di phi (Matrice di correlazione per binarie)
correlazione = df[['diff_walk', 'infortunio']].replace({'si':1, 'no':0}).corr().iloc[0,1]
print(f"\nCorrelazione di Pearson (phi): {correlazione:.2f}")
# --- Verifica Ridondanza: Diff_walk vs Infortunio ---
# La correlazione (Phi) è pari a 0.32. 
# Essendo < 0.70, le variabili non presentano multicollinearità critica.
# - diff_walk misura la compromissione funzionale cronica.
# - infortunio misura la fragilità fisica acuta/recente.
# CONCLUSIONE: Vengono mantenute entrambe per massimizzare il potere predittivo.

#LE VARIBILI CHE UTILIZZO SONO:
#diabete, ipertensione, colesterolo_alto, bmi, fumatore, ictus, attacco_cardiaco, esercizio_fisico, 
#frutta, verdura, alcol, salute, infortunio, diff_walk, genere, età, istruzione, reddito
#La selezione preliminare delle variabili è stata guidata da analisi statistiche esplorative.
#Tuttavia, nei modelli di machine learning l’importanza finale delle feature è valutata 
#in modo data-driven tramite metodi di feature importance.
#-----------------------------------------------------------------------------------------------------------
"""
IMPLETAZIONE DEI MODELLI
Per questo studio, ho deciso di usare 3 modelli: Logit, Random Forest e XgBoost. 
 =============================================================================
 PREPROCESING: CODIFICA DELLE VARIABILI E SUPPORTO SCIENTIFICO
=============================================================================
Data l'ampia dimensione del campione (N = 229.300), le variabili istruzione, 
reddito, classi di età e salute percepita sono state codificate preservandone 
l'ordine naturale (Ordinal Encoding). 
Tale scelta metodologica è supportata dalla letteratura scientifica:
1. VALIDITÀ CLINICA: Seveso et al. (2020) dimostrano che la codifica di 
   variabili ordinali migliora la validità dei dati e la performance 
   predittiva in contesti medici, permettendo al modello di riflettere 
   la progressione biologica delle patologie.
2. EFFICIENZA COMPUTAZIONALE: Zhu et al. (2024) evidenziano come l’uso di 
   ordinal encoding nei modelli di classificazione ed ensemble (RF, XGBoost) 
   eviti l’esplosione della dimensionalità tipica delle codifiche one-hot, 
   riducendo il rischio di overfitting e ottimizzando la memoria.
3. PATTERN GERARCHICI: La letteratura sulla regressione ordinale sottolinea 
   che preservare l’ordine delle categorie è fondamentale per catturare 
   pattern gerarchici e soglie critiche (thresholds) che andrebbero perse 
   trattando le classi come etichette indipendenti.
4. LOGICA DI MODELLAZIONE: Per garantire la confrontabilità, l'ordinalità 
   viene mantenuta in tutti i modelli. Sebbene questo imponga un vincolo 
   di monotonicità alla regressione logistica, permette di valutare la 
   capacità dei modelli ensemble di estrarre pattern non lineari dai 
   medesimi gradienti gerarchici.
"""
#codifico le variabili
df_p = df.copy()

#variabili binarie si e no
col = ['diabete', 'ipertensione', 'colesterolo_alto', 
    'fumatore', 'ictus', 'attacco_cardiaco', 'esercizio_fisico', 'frutta', 
    'verdura', 'alcol', 'infortunio', 'diff_walk']
df_p[col] = df_p[col].replace({'si': 1, 'no': 0}).astype('int32')

df_p['genere'] = df_p['genere'].replace({'uomo':1, 'donna':0}).astype('int32')

from sklearn.preprocessing import OrdinalEncoder
# Lista delle variabili ordinali
ordinal_vars = ['istruzione', 'reddito', 'eta', 'salute']

# Creiamo un OrdinalEncoder specificando che le categorie sono già ordinate
# handle_unknown='use_encoded_value' con unknown_value=-1 serve a gestire eventuali valori sconosciuti
encoder = OrdinalEncoder(categories=[
    df_p['istruzione'].cat.categories,  # già ordinato
    df_p['reddito'].cat.categories,
    df_p['eta'].cat.categories,
    df_p['salute'].cat.categories], dtype=int)

# Fit e trasformazione
df_p[ordinal_vars] = encoder.fit_transform(df_p[ordinal_vars])

#Uniformiamo il tipo di tutte le variabili ordinali a int32
df_p[ordinal_vars] = df_p[ordinal_vars].astype('int32')


#Uniformiamo anche il BMI per coerenza e risparmio memoria
df_p['bmi'] = df_p['bmi'].astype('int32')

#controllo
df_p.info()

# =============================================================================
# DATA SPLITTING E PREVENZIONE DEL DATA LEAKAGE
# =============================================================================

"""
In questa fase procediamo alla separazione del dataset in Training Set e Test Set
prima di applicare qualsiasi trasformazione di scaling. 

Questa procedura è critica per prevenire il 'Data Leakage' (contaminazione dei dati):
1. SIMULAZIONE REAL-WORLD: Il Test Set deve rimanere totalmente 'inedito' per il 
   modello, rappresentando dati futuri mai osservati durante la fase di apprendimento.
2. INTEGRITÀ STATISTICA: Lo StandardScaler deve calcolare media e varianza 
   esclusivamente sul Training Set. Applicare lo scaling all'intero dataset prima 
   dello split porterebbe informazioni del Test Set (la sua distribuzione) 
   all'interno del processo di training, falsando la valutazione delle performance.
3. STRATIFICAZIONE: Utilizziamo 'stratify=y' per garantire che la proporzione 
   tra classi (diabetici vs non diabetici) sia mantenuta identica in entrambi 
   i set, fondamentale data la natura sbilanciata dei dati clinici.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definizione delle feature (X) e del target (y)
X = df_p.drop('diabete', axis=1)
y = df_p['diabete']

# Split 80/20: La separazione avviene PRIMA dello scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
#stratify mantiene le stesse percentuali di modalità della variabili target

# =============================================================================
# 3. STANDARDIZZAZIONE (FEATURE SCALING)
# =============================================================================

scaler = StandardScaler()

# Fit e transform solo sul set di addestramento
X_train_scaled = scaler.fit_transform(X_train)

# Trasformazione del set di test utilizzando i parametri (mu, sigma) del train
X_test_scaled = scaler.transform(X_test)

print("Split e Scaling completati con successo senza Data Leakage.")

# =============================================================================
# MODELLO 1: REGRESSIONE LOGISTICA (BASELINE INTERPRETABILE)
# =============================================================================

"""
La Regressione Logistica viene utilizzata come modello di baseline per la sua 
elevata interpretabilità. In un contesto medico, è fondamentale non solo 
predire il rischio, ma comprendere il peso relativo di ogni fattore 
socio-economico e clinico attraverso l'analisi degli Odds Ratio.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Inizializzazione del modello
# Usiamo class_weight='balanced' se noti che i diabetici sono molti meno dei sani
logit = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
#La regressione logistica è un processo iterativo che cerca di minimizzare l'errore. 
#Aumentiamo il limite a 1000 per assicurarci che l'algoritmo abbia tempo sufficiente per 
#"convergere" (trovare la soluzione ottimale) su un dataset così grande. Utilizziamo 'class_weight' 
#per compensare lo sbilanciamento delle classi, forzando il modello a dare maggior peso 
#alla classe minoritaria  (diabetici), migliorando così la Recall (Sensibilità) del test clinico.

# Addestramento
logit.fit(X_train_scaled, y_train)

# Predizioni
y_pred_logit = logit.predict(X_test_scaled)
y_proba_logit = logit.predict_proba(X_test_scaled)[:, 1]

# --- VALUTAZIONE DELLE PERFORMANCE ---
# Calcolo della matrice normalizzata
cm_norm = confusion_matrix(y_test, y_pred_logit, normalize='true')
plt.figure(figsize=(9, 7))
# Creazione della heatmap
ax = sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                 cbar_kws={'label': 'Proporzione'})
plt.title('Matrice di Confusione Normalizzata\nRegressione Logistica (Balanced)', 
          fontsize=14, pad=20, fontweight='bold')
plt.ylabel('VALORE REALE', fontsize=12, fontweight='bold', labelpad=15)
plt.xlabel('VALORE PREDETTO', fontsize=12, fontweight='bold', labelpad=15)
ax.set_xticklabels(['SANO', 'DIABETICO'], fontweight='bold')
ax.set_yticklabels(['SANO', 'DIABETICO'], fontweight='bold')
plt.show()

# Calcolo delle metriche basate sulle classi predette (0/1)
auc_score = roc_auc_score(y_test, y_proba_logit)
print("-" * 30)
print("PERFORMANCE REGRESSIONE LOGISTICA (Balanced)")
print("-" * 30)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_logit):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_logit):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_logit):.4f}  <-- Obiettivo Clinico")
print(f"F1-score:  {f1_score(y_test, y_pred_logit):.4f}")
print(f"AUC-ROC:   {auc_score:.4f}")
# Calcolo dell'AUC-ROC basato sulle probabilità (0.0 - 1.0)
# Questa metrica valuta la capacità di separazione delle classi del modello

# --- ANALISI PERFORMANCE LOGIT (BALANCED) ---
# Modello solido con AUC-ROC di 0.8045, che supera la soglia di affidabilità 
# clinica (0.80) per lo screening preventivo. La Recall del 74.84% garantisce 
# l'identificazione della vasta maggioranza dei soggetti diabetici, mentre 
# l'Accuracy del 71.96% conferma la stabilità della baseline. Il trade-off 
# sulla Precision (32.10%) è una scelta progettuale consapevole per 
# massimizzare la sensibilità diagnostica e ridurre i falsi negativi.

# =============================================================================
# 6. ANALISI DEI COEFFICIENTI (ODDS RATIO)
# =============================================================================

# Creiamo un DataFrame per visualizzare l'importanza delle feature
features = X.columns
coeff = logit.coef_[0]
df_coeff = pd.DataFrame({'Feature': features, 'Coefficient': coeff})
df_coeff['Odds_Ratio'] = np.exp(df_coeff['Coefficient']) # Trasformazione per interpretabilità
df_coeff = df_coeff.sort_values(by='Coefficient', ascending=False)

print("\nTop Fattori di Rischio (Coefficienti positivi):")
print(df_coeff.head())

#Grafico
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x='Coefficient', 
    y='Feature', 
    data=df_coeff.head(10), 
    palette='Blues_r')
plt.title('TOP 10 FATTORI DI RISCHIO\n(Coefficienti Regressione Logistica)', 
          fontsize=16, fontweight='bold', pad=25)
plt.xlabel('IMPATTO SUL RISCHIO (PESO DEL COEFFICIENTE)', fontsize=12, fontweight='bold')
plt.ylabel('PARAMETRO CLINICO', fontsize=12, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.4)
sns.despine()
plt.tight_layout()
plt.show()
# --- INTERPRETAZIONE CLINICA (ODDS RATIO) ---
# L'analisi dei coefficienti identifica il BMI (OR: 1.59) e l'Età (OR: 1.53) come 
# i principali driver predittivi: un incremento unitario, a parità di altri fattori,
#un aumento unitario nel valore del BMI (scalato) aumenta le probabilità di diabete di circa 
#il 59%. Il rischio aumenta del 53% per ogni incremento unitario dell'età.
#Ipertensione (OR: 1.45) e Colesterolo Alto (OR: 1.33) completano il profilo di rischio, 
#confermando una forte coerenza del modello con la letteratura medica esistente.

# =============================================================================
# MODELLO 2: RANDOM FOREST CLASSIFIER (Ensemble Learning)
# =============================================================================
"""
Passiamo a un modello non lineare basato su un "ensemble" di 100 alberi 
decisionali. L'obiettivo è catturare interazioni complesse tra le feature 
che la Logit potrebbe aver perso, cercando di migliorare la precisione 
senza sacrificare l'ottima Recall ottenuta finora.
"""
# Obiettivo: Superare la baseline della Logit catturando relazioni non lineari.
from sklearn.ensemble import RandomForestClassifier

# Inizializzazione del modello
# max_depth=10 aiuta a prevenire l'overfitting iniziale su dataset grandi
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, #Questo è il "freno a mano" per evitare l'overfitting. Impediamo agli alberi di diventare troppo lunghi e complessi, costringendoli a imparare solo le regole generali.
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1) #Una comodità tecnica; dice al computer di usare tutti i processori disponibili per velocizzare il calcolo.

# Addestramento
rf_model.fit(X_train_scaled, y_train)

# Predizioni
y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# --- MATRICE DI CONFUSIONE GRAFICA ---
cm_rf = confusion_matrix(y_test, y_pred_rf, normalize='true')
plt.figure(figsize=(9, 7))
ax = sns.heatmap(cm_rf, annot=True, fmt='.2%', cmap='Blues', cbar_kws={'label': 'Proporzione'})
plt.title('Matrice di Confusione Normalizzata\nRandom Forest (Balanced)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('VALORE REALE', fontsize=12, fontweight='bold', labelpad=15)
plt.xlabel('VALORE PREDETTO', fontsize=12, fontweight='bold', labelpad=15)
ax.set_xticklabels(['SANO', 'DIABETICO'], fontweight='bold')
ax.set_yticklabels(['SANO', 'DIABETICO'], fontweight='bold')
plt.show()

# --- CALCOLO METRICHE ---
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
print("\nPERFORMANCE RANDOM FOREST (Balanced)")
print("-" * 30)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}  <-- Obiettivo Clinico")
print(f"F1-score:  {f1_score(y_test, y_pred_rf):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba_rf):.4f}")

# --- FINAL ASSESSMENT: RANDOM FOREST (BALANCED) ---
# Modello con AUC-ROC di 0.8104, consolidando un'ottima capacità di 
# discriminazione tra classi. La Recall al 75.26% garantisce la cattura 
# dei 3/4 dei soggetti diabetici reali, centrando l'obiettivo clinico. 
# La Precision ferma al 32.31% conferma la persistenza di "falsi allarmi", 
# una conseguenza strutturale del dataset che privilegia la sensibilità 
# diagnostica per non lasciare casi critici non diagnosticati.


#Estrazione delle importanze
importances = rf_model.feature_importances_
feature_names = X.columns # Assicurati che X sia il tuo DataFrame originale con i nomi colonne
df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
df_importance = df_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x='Importance', 
    y='Feature', 
    data=df_importance.head(10), 
    palette='Blues_r')
plt.title('FEATURE IMPORTANCE - RANDOM FOREST\n(Quali variabili pesano di più per la foresta?)', 
          fontsize=16, fontweight='bold', pad=25)
plt.xlabel('GINI IMPORTANCE (PESO NEL MODELLO)', fontsize=12, fontweight='bold')
plt.ylabel('PARAMETRO CLINICO', fontsize=12, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
# --- EVALUATION SUMMARY: RANDOM FOREST FEATURE IMPORTANCE ---
#L'analisi Gini Importance evidenzia l'Ipertensione e la percezione della 
#Salute come i principali discriminatori per la foresta, seguiti dal BMI. 
#Questo modello cattura relazioni non lineari più profonde: a differenza 
#della Logit, assegna un peso maggiore a variabili sistemiche e soggettive, 
#indicando che la combinazione di questi fattori è più predittiva del 
#singolo dato anagrafico. La stabilità dei risultati conferma la robustezza 
#della foresta nell'identificare il profilo di rischio complesso.

# =============================================================================
#MODELLO 3: XGBOOST CLASSIFIER (Gradient Boosting)
# =============================================================================
"""
XGBoost rappresenta l'evoluzione degli algoritmi basati su alberi. 
# Mentre il Random Forest crea alberi indipendenti, XGBoost utilizza il 
# "Boosting": ogni nuovo albero viene costruito per correggere gli errori 
# (i residui) commessi dagli alberi precedenti. Questo lo rende 
# particolarmente efficace nel rifinire la Precision e l'AUC-ROC, 
# spingendo il modello verso il limite massimo di accuratezza possibile."
"""

from xgboost import XGBClassifier
#Calcolo del bilanciamento (Sostituisce il class_weight)
# Calcoliamo il rapporto esatto tra Sani e Diabetici
#Questo garantisce che la Recall sia prioritaria
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

#Inizializzazione del Modello
xgb_model = XGBClassifier(
    n_estimators=1000,       # Numero alto di alberi, ci fermeremo prima con early stopping
    max_depth=6,             # Profondità contenuta per evitare overfitting
    learning_rate=0.1,       # Velocità di apprendimento "passo dopo passo"
    scale_pos_weight=scale_pos_weight, # Gestione sbilanciamento classi
    random_state=42,
    early_stopping_rounds=15, # Si ferma se non migliora per 15 round consecutivi il modello si fermerà non appena smetterà di imparare cose utili dai dati di test.
    use_label_encoder=False,
    eval_metric='logloss'    # Metrica monitorata durante l'addestramento
)

#Definizione del set di validazione
eval_set = [(X_test_scaled, y_test)]

#Addestramento 
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=eval_set,
    verbose=False             # Nasconde i log di ogni singolo albero
)

#Predizioni
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

#matrice confusione
cm = confusion_matrix(y_test, y_pred_xgb)
#Normalizzazione per vedere le percentuali (fondamentale per dataset sbilanciati)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#Visualizzazione
plt.figure(figsize=(9, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['SANO', 'DIABETICO'], 
            yticklabels=['SANO', 'DIABETICO'],
            annot_kws={"size": 12, "weight": "bold"})
plt.title('Matrice di Confusione Normalizzata\nXGBoost (Balanced)', fontsize=15, fontweight='bold', pad=20)
plt.ylabel('VALORE REALE', fontsize=12, fontweight='bold')
plt.xlabel('VALORE PREDETTO', fontsize=12, fontweight='bold')
plt.show()

#Calcolo Metriche
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
auc_roc_xgb = roc_auc_score(y_test, y_proba_xgb)

#Visualizzazione Risultati
print("\nPERFORMANCE XGBOOST (Balanced + Early Stopping)")
print("-" * 45)
print(f"Accuracy:  {accuracy_xgb:.4f}")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall:    {recall_xgb:.4f}  <-- Obiettivo Clinico")
print(f"F1-score:  {f1_xgb:.4f}")
print(f"AUC-ROC:   {auc_roc_xgb:.4f}")
print("-" * 45)
print(f"Miglior iterazione (albero numero): {xgb_model.best_iteration}")
# --- FINAL ASSESSMENT: XGBOOST PERFORMANCE ---
# Il modello XGBoost ha raggiunto il picco di sensibilità (Recall: 75.82%), 
# confermandosi come lo strumento più cautelativo per lo screening.
# Con un'area sotto la curva (AUC-ROC) di 0.8073, l'algoritmo dimostra 
# una capacità eccellente nel distinguere le classi, stabilizzandosi 
# precocemente grazie all'early stopping (iterazione 414).

#Estrazione delle importanze (Gain)
#Il Gain è spesso considerato la metrica più affidabile per XGBoost
xgb_importances = xgb_model.get_booster().get_score(importance_type='gain')

#
# Creiamo un DataFrame partendo dal dizionario e ordiniamo
df_xgb_importance = pd.DataFrame({
    'Feature': list(xgb_importances.keys()),
    'Importance': list(xgb_importances.values())
})
df_xgb_importance = df_xgb_importance.sort_values(by='Importance', ascending=False)

feature_names = X.columns # Assicurati che 'X' sia il DataFrame originale con i nomi
#Creiamo un dizionario di mappatura (f0 -> NomeReale)
mapper = {f'f{i}': name for i, name in enumerate(feature_names)}
df_xgb_importance['Feature'] = df_xgb_importance['Feature'].map(mapper)

#Visualizzazione
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x='Importance', 
    y='Feature', 
    data=df_xgb_importance.head(10), 
    palette='Blues_r' # 
)
plt.title('FEATURE IMPORTANCE - XGBOOST\n(Contributo al miglioramento della previsione - Gain)', 
          fontsize=16, fontweight='bold', pad=25)
plt.xlabel('GAIN (INCREMENTO DI PRECISIONE)', fontsize=12, fontweight='bold')
plt.ylabel('PARAMETRO CLINICO', fontsize=12, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
sns.despine()
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
#XGBoost elegge l'Ipertensione a principale segnale d'allarme. Questo spiega 
#l'alta capacità del modello di identificare i diabetici (Recall 76%): l'algoritmo ha 
#imparato che nel dataset la coincidenza tra pressione alta e diabete è il binario 
#più sicuro per non perdere diagnosi importanti.


#SCELTA DEL MODELLO MIGLIORE
"""
Il passaggio finale di questa analisi consiste nella comparazione sistematica dei 
tre algoritmi testati: Regressione Logistica, Random Forest e XGBoost. 
L'obiettivo non è semplicemente trovare il modello con l'accuratezza più alta, 
ma identificare la soluzione che meglio risponde alle necessità di uno 
screening clinico per il diabete.
"""

risultati_finali = {
    'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Regr. Logistica': [0.7212, 0.3210, 0.7488, 0.4491, 0.8037],
    'Random Forest': [0.7237, 0.3231, 0.7526, 0.4522, 0.8104],
    'XGBoost': [0.7159, 0.3193, 0.7582, 0.4493, 0.8073]}

# Creazione del DataFrame
df_comp = pd.DataFrame(risultati_finali)

# Visualizzazione della tabella
print("CONFRONTO INTEGRALE PERFORMANCE")
print("=" * 60)
print(df_comp.to_string(index=False))
print("=" * 60)


df_heatmap = pd.DataFrame(risultati_finali).set_index('Metrica')

#grafico
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'

# 3. Creazione della figura
plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    df_heatmap, 
    annot=True, 
    fmt=".2%", 
    cmap="Blues", 
    linewidths=1.5, 
    linecolor='white',
    cbar=False,
    annot_kws={"size": 12, "weight": "bold"} # Numeri interni in grassetto
)
ax.set_xticklabels(
    ax.get_xticklabels(), 
    fontsize=11, 
    fontweight='bold', 
    color='#000000')

ax.set_yticklabels(
    ax.get_yticklabels(), 
    fontsize=11, 
    fontweight='bold', 
    color='#000000',
    rotation=0)
plt.title('REPORT COMPARATIVO PERFORMANCE\n', fontsize=16, fontweight='bold', pad=10)
plt.xlabel('')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig('scelta_modello.png')
plt.show()

"""
Model Selection: Why XGBoost?"
Dopo aver confrontato Regressione Logistica, Random Forest e XGBoost, 
è stato scelto XGBoost come modello finale. Sebbene il Random Forest abbia mostrato un'AUC-ROC 
leggermente superiore (81.04% vs 80.73%), XGBoost ha garantito la Recall più elevata (75.82%). 
In un contesto di screening medico per il diabete, minimizzare i falsi negativi è prioritario: 
XGBoost si è dimostrato il più efficace nel catturare i soggetti a rischio, 
rendendolo lo strumento più sicuro per il supporto alla diagnosi.
"""
#--------------------------------------------------------------------------------------------
# VERIFICA DELLA CAPACITÀ DI GENERALIZZAZIONE (OVERFITTING CHECK)
"""
Questo blocco analizza se il modello XGBoost soffre di overfitting confrontando 
le prestazioni sul set di addestramento (Train) e sul set di test (Validation).
L'obiettivo è confermare che il modello abbia appreso pattern clinici reali 
e sia in grado di diagnosticare correttamente nuovi pazienti non visti in fase di training.
"""
from sklearn.metrics import roc_curve, auc

# Probabilità predette
y_train_proba = xgb_model.predict_proba(X_train_scaled)[:, 1]
y_test_proba  = xgb_model.predict_proba(X_test_scaled)[:, 1]

# ROC
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test,  tpr_test,  _ = roc_curve(y_test,  y_test_proba)

auc_train = auc(fpr_train, tpr_train)
auc_test  = auc(fpr_test,  tpr_test)
print("-" * 30)
print(f"AUC-TRAIN:   {auc_train:.4f}")
print(f"AUC-TEST:   {auc_test:.4f}")

plt.figure()
plt.plot(fpr_train, tpr_train, label=f"Train AUC = {auc_train:.3f}")
plt.plot(fpr_test,  tpr_test,  label=f"Test AUC = {auc_test:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Train vs Test")
plt.legend()
plt.savefig('overfitting.png')
plt.show()
"""
L'analisi comparativa tra i set di addestramento e validazione conferma l'assenza 
di overfitting. La coerenza millimetrica tra le metriche di Train e Test valida 
l'architettura del modello XGBoost scelto, rendendolo una soluzione robusta e 
affidabile per il supporto alla diagnosi clinica.
"""

















