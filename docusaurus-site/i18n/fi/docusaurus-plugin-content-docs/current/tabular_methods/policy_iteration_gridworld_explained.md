# Käytännön iteraatio GridWorldille 🗺️

## Mikä se on?

Kuvittele, että pelaat lautapeliä **4×4-ruudukolla** (kuten pienellä shakkilaudalla). Aloitat sinä
toisessa kulmassa ja täytyy päästä toiseen kulmaan. Jokainen askel maksaa 1 pisteen (et halua
tuhlata askeleita!), ja tavoitteen saavuttaminen ei ansaitse mitään ylimääräistä – haluat vain päästä perille
mahdollisimman nopeasti.

**Käytännön iterointi** on tapa, jolla tietokone selvittää **parhaat liikkeet jokaiselle ruudulle**
laudalla – kaikki kerralla!

---

## Suuri idea: kaksi askelta, yli ja yli

Ajattele sitä kuin huoneen puhdistamista avustajan kanssa:

1. **Vaihe 1 – Selvitä, kuinka hyvä kukin ruutu on (käytännön arviointi)**
   Auttajasi kävelee jokaisen ruudun ympäri ja kirjoittaa muistiin: "Jos noudatan nykyistä suunnitelmaa, miten
   kestääkö monta askelta päästäkseni uloskäynnille täältä?" He tekevät tämän uudestaan ja uudestaan, kunnes
   numerot lakkaavat muuttumasta.

2. **Vaihe 2 – Paranna suunnitelmaa (käytännön parantaminen)**
   Nyt katsot jokaista ruutua ja kysyt: "Onko olemassa parempaa suuntaa, johon voisin mennä täältä?"
   Jos kyllä, päivitä suunnitelma!

Toista vaiheita 1 ja 2, kunnes suunnitelma lakkaa muuttumasta – se on **optimaalinen käytäntö**!

**Tosielämän esimerkki:** Kuvittele, että löydät nopeimman reitin kouluun. Ensin arvaat reitin
ja ajasta se (vaihe 1). Sitten katsot jokaista kadunkulmaa ja kysyt "onko sieltä pikakuvaketta
täällä?" (Vaihe 2). Päivität reittisi ja toistat, kunnes et löydä enää pikakuvakkeita!

---

## Mitä koodimme löysi

4×4 GridWorldissämme on kaksi päätetilaa (kulmaa), ja agentti maksaa -1 askelta kohti.
Käytännön iteraatio lähentyi vain **4 kierroksella** (139 arviointipyyhkäisyä):

```
State Values V(s):       Optimal Policy:
 0.0  -1.0  -1.9  -2.7    T   ←   ←   ↓
-1.0  -1.9  -2.7  -1.9    ↑   ↑   ↑   ↓
-1.9  -2.7  -1.9  -1.0    ↑   ↑   ↓   ↓
-2.7  -1.9  -1.0   0.0    ↑   →   →   T
```

**Arvot ovat täysin järkeviä!** Päätteen vieressä olevien ruutujen arvo on -1 (yhden askeleen päässä).
Kahden askeleen päässä olevien ruutujen arvo on -1,9 (= -1 + 0,9 × -1) ja niin edelleen.

---

## Esimerkkejä tosielämästä

- **GPS-navigointi**: Paras käännös kartan *jokaisessa* risteyksessä.
- **Hissin ohjaus**: Mihin kerrokseen hissin tulisi mennä, kun sillä on useita pyyntöjä?
- **Tehdasrobotti**: Tehokkaimman polun suunnittelu varastoruudukon ympäri.

---

## Avainsanat muistaa

- **Käytäntö**: Suunnitelma – mitä toimia kussakin osavaltiossa tulee tehdä
- **Arvofunktio V(s)**: Kuinka hyvä on olla tilassa s (korkeampi = lähempänä tavoitetta)
- **Käytännön arviointi**: Lasketaan, kuinka hyvä nykyinen suunnitelma on
- **Käytännön parantaminen**: Suunnitelman parantaminen arvofunktion avulla
- **Optimaalinen käytäntö**: Paras mahdollinen suunnitelma – ei voi enää parantaa

Suuri idea: **Sinun ei tarvitse kokeilla kaikkia mahdollisia suunnitelmia! Jatka vain virran parantamista
yksi, ja löydät parhaan suunnitelman vain muutamalla kierroksella.**
