# Doelgericht beleid

## Het grote idee: één beleid om ze allemaal te regeren {#the-big-idea-one-policy-to-rule-them-all}

Stel je voor dat je een bezorger bent. Je hebt niet voor elk adres een compleet andere vaardigheden nodig. U weet hoe u moet autorijden, een kaart moet lezen en door het verkeer moet navigeren. U hoeft alleen maar de *bestemming van vandaag* in te voeren en te gaan.

Een **doelgericht beleid** werkt op dezelfde manier. In plaats van één agent op te leiden die slechts één vast doel kan bereiken, trainen we één enkele agent die elk doel als input accepteert en uitzoekt hoe hij daar kan komen.

---

## Hoe het verschilt van standaard RL {#how-it-differs-from-standard-rl}

In standaard RL (zoals behandeld in de eerdere fasen van het curriculum) is de beloningsfunctie ingebakken: "bereik cel (7, 7), krijg +1." De agent leert precies één ding: hoe hij *die* cel kan bereiken.

Bij doelgerichte RL hangt de beloning af van het feit of de agent *ongeacht het doel dat hem deze keer werd gegeven* bereikt. Het beleid leert:

> **"Wat moet ik doen, gezien waar ik ben en waar ik wil zijn?"**

Het doel reist *met* de agent, zoals een bestemming die in een navigatie-app wordt getypt.

---

## Het schaarse beloningsprobleem {#the-sparse-reward-problem}

Hier is het addertje onder het gras: leren van schaarse beloningen (slechts +1 bij het doel, 0 overal elders) is brutaal moeilijk. De meeste pogingen mislukken: de agent dwaalt willekeurig rond, komt nooit het doel tegen, en het netwerk krijgt niets nuttigs om van te leren.

Stel je voor dat je probeert geblinddoekt een pijltje te leren gooien. Je gooit duizend keer en mist altijd. Na duizend mislukkingen heb je nog steeds geen idee hoe "een goede worp" voelt.

Dit is waar **Hindsight Experience Replay (HER)** in beeld komt.

---

## Ervaring achteraf opnieuw afspelen: voorwaarts falen {#hindsight-experience-replay-failing-forward}

De truc van HAAR is prachtig eenvoudig. Na een mislukte aflevering vraagt ​​HAAR:

> *"Ook al heb je je doel niet bereikt... waar ben je eigenlijk terechtgekomen?"*

Vervolgens wordt **diezelfde aflevering** herhaald, maar wordt gedaan alsof de werkelijke uiteindelijke positie van de agent **de hele tijd** het doel was. Plots wordt een mislukte aflevering een succesvolle aflevering – voor een ander doel.

Het is als een mislukte basketbalspeler die steeds maar op de hoepel schiet en mist. HAAR zou zeggen: "Oké, je raakt elke keer de linkermuur. Gefeliciteerd - je bent geweldig in het raken van de linkermuur! Laten we die worpen registreren als succesvolle pogingen om de linkermuur te raken." Na verloop van tijd bouwt de speler vaardigheid op in het raken van *elk* doelwit, en brengt dat uiteindelijk over naar de echte ring.

Dit verandert duizenden "mislukkingen" in een rijke bibliotheek van *succesvolle* navigatie naar veel verschillende plekken. De agent leert ze allemaal te bereiken, wat zich generaliseert naar het echte doelwit.

---

## De analogie uit het echte leven: peuters leren blokken op elkaar te stapelen {#the-real-life-analogy-toddler-learning-to-stack-blocks}

Een peuter die probeert een blok in een emmer te stoppen, mist voortdurend. Maar elke "misser" zorgt ervoor dat het blok *ergens* terechtkomt. Als je elke misser herhaalt als "je probeerde hem *precies daar* neer te zetten - en dat is gelukt!", bouwt de peuter de fijne motoriek over de hele tafel op. Binnenkort kunnen ze overal een blok plaatsen, ook in de emmer.

---

## Wat onze code doet {#what-our-code-does}

Het script `goal_conditioned_policy.py` draait in een **7x7 doolhof** met muren. Aan het begin van elke aflevering wordt een willekeurige doelcel gekozen. De agent moet het vinden.

Het beleid heeft bij elke stap twee inputs nodig:
1. Waar de agent momenteel is
2. Waar het heen wil

Na elke aflevering (al dan niet succesvol) genereert HER verschillende extra synthetische "successen" door de feitelijk bezochte posities opnieuw te bestempelen als alternatieve doelen.

De training duurt 3000 afleveringen met een afnemende verkenningssnelheid. De agent onderzoekt eerst meer en vertrouwt daarna steeds meer op wat hij heeft geleerd.

---

## Wat de grafieken laten zien {#what-the-charts-show}

![Doelafhankelijke beleidsresultaten](outputs/goal_conditioned_policy.png)

**Links: succespercentage tijdens training:** Elke aflevering is een succes (het doel bereikt) of een mislukking. De curve stijgt gestaag naarmate de universele navigatievaardigheid van de agent verbetert. Tegen het einde bereikt de agent bijna elke keer een doel.

**Rechts: Heatmap voor succespercentage van doelen:** Na de training testen we de agent op elke mogelijke doelcel en kleuren we elke cel in op basis van hoe vaak de agent deze bereikt. Groen betekent dat de agent die plek betrouwbaar bereikt; rood betekent dat het nog steeds moeite heeft. Een goed opgeleide agent toont overwegend groen in het hele doolhof.

---

## Waar dit in de echte wereld verschijnt {#where-this-shows-up-in-the-real-world}

| Sollicitatie | Het "doel" |
|-------------|------------|
| Robotarm reikt | Doel 3D-positie |
| Zelfrijdende auto | GPS-coördinaat |
| Assistent taalmodel | Gebruiker instructie |
| Niet-spelerkarakter van videogame | Elk waypoint op de kaart |

Doelgericht beleid is een van de belangrijkste bouwstenen voor HIRO (hiërarchische RL met subdoelen) – de manager op hoog niveau kiest een subdoel, en de werknemer op laag niveau is precies dit soort doelgericht beleid.

---

## Samenvatting van één zin {#one-sentence-summary}

> **Een doelgericht beleid is één agent die naar elke bestemming kan navigeren – en HAAR maakt leren van mislukkingen mogelijk door te doen alsof elk gemist schot gericht was op de plek waar het terechtkwam.**