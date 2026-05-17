# Beleidsiteratie voor GridWorld 🗺️

## Wat is het?

Stel je voor dat je een bordspel speelt op een **4×4 raster** (zoals een klein schaakbord). Jij begint
in de ene hoek en moet de andere hoek bereiken. Elke stap kost 1 punt (dat wil je niet
om stappen te verspillen!), en het bereiken van het doel levert je niets extra’s op; je wilt er gewoon komen
zo snel mogelijk.

**Beleidsiteratie** is hoe een computer de **beste zetten voor elk vierkant** berekent
op het bord – allemaal tegelijk!

---

## Het grote idee: twee stappen, keer op keer

Zie het als het schoonmaken van je kamer met een helper:

1. **Stap 1 — Zoek uit hoe goed elk vierkant is (beleidsevaluatie)**
   Je helper loopt elk vierkant rond en schrijft op: "Als ik het huidige plan volg, hoe
   Hoeveel stappen zal ik nodig hebben om vanaf hier de uitgang te bereiken?" Ze doen dit keer op keer totdat
   de cijfers veranderen niet meer.

2. **Stap 2 — Verbeter het plan (beleidsverbetering)**
   Nu kijk je naar elk vierkant en vraag je: "Is er een betere richting die ik vanaf hier kan inslaan?"
   Zo ja, update het plan!

Herhaal stap 1 en 2 totdat het plan niet meer verandert — dat is het **optimale beleid**!

**Voorbeeld uit de praktijk:** Stel je voor dat je de snelste route naar school vindt. Eerst raad je een route
en tijd (stap 1). Dan kijk je op elke straathoek en vraag je ‘is daar een kortere weg vandaan?
hier?" (Stap 2). Je werkt je route bij en herhaalt dit totdat je geen snelkoppelingen meer kunt vinden!

---

## Wat onze code heeft gevonden

Onze 4×4 GridWorld heeft twee eindtoestanden (hoeken) en de agent betaalt -1 per stap.
Beleidsiteratie kwam samen in slechts **4 ronden** (in totaal 139 evaluatierondes):

```
State Values V(s):       Optimal Policy:
 0.0  -1.0  -1.9  -2.7    T   ←   ←   ↓
-1.0  -1.9  -2.7  -1.9    ↑   ↑   ↑   ↓
-1.9  -2.7  -1.9  -1.0    ↑   ↑   ↓   ↓
-2.7  -1.9  -1.0   0.0    ↑   →   →   T
```

**De waarden zijn volkomen logisch!** Vierkantjes naast een terminal hebben waarde -1 (één stap verwijderd).
Vierkantjes op twee stappen afstand hebben de waarde -1,9 (= -1 + 0,9 × -1), enzovoort.

---

## Voorbeelden uit het echte leven

- **GPS-navigatie**: het uitzoeken van de beste afslag op *elk* kruispunt op de kaart.
- **Liftbediening**: Naar welke verdieping moet de lift gaan als er meerdere verzoeken zijn?
- **Fabrieksrobot**: Plan het meest efficiënte pad rond een magazijnraster.

---

## Sleutelwoorden om te onthouden

- **Beleid**: het plan – welke actie moet worden ondernomen in elke staat
- **Waardefunctie V(s)**: Hoe goed het is om in staat s te zijn (hoger = dichter bij doel)
- **Beleidsevaluatie**: berekenen hoe goed het huidige plan is
- **Beleidsverbetering**: het plan verbeteren met behulp van de waardefunctie
- **Optimaal beleid**: het best mogelijke plan — kan niet verder worden verbeterd

Het grote idee: **Je hoeft niet elk mogelijk plan uit te proberen! Blijf gewoon de stroming verbeteren
één, en je vindt het beste plan in een paar rondes.**