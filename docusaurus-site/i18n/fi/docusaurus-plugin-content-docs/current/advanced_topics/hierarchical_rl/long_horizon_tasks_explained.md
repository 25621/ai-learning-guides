# Pitkän horisontin tehtävät

## Suuri idea: Kun palkinto on hyvin kaukana {#the-big-idea-when-the-reward-is-very-far-away}

Kuvittele, että olet kokki, joka yrittää oppia uuden reseptin puhtaasti maistamalla viimeistä ruokaa. Noudatat 40 vaihetta – pilko, hauduta, mausta, hauduta, lautaset – mutta saat palautetta vasta aivan lopussa: "Liian suolaista." Mikä 40 vaiheesta aiheutti ongelman? Sinulla ei ole aavistustakaan.

Tämä on **pitkän horisontin ongelma**: kun palkintosignaali erotetaan sen aiheuttamista päätöksistä kymmenillä (tai sadoilla) askelilla, oppimisesta tulee erittäin vaikeaa.

---

## Miksi tasaiset agentit kamppailevat {#why-flat-agents-struggle}

Tasainen RL-agentti (kuten vaiheen 3 DQN-agentit) yrittää oppia jokaisen vaiheen arvon kerralla. Lyhyissä tehtävissä – tasapainota sauva, vältä seinää – tämä toimii hyvin. Palkkio saapuu nopeasti, ja agentti voi yhdistää syyn ja seurauksen.

Mutta pitkässä tehtävässä – kerätä avain, avaa se sitten ovi ja poistu sitten sokkelosta – agentin on:

1. Kompastele avaimeen (onnekas!)
2. Muista, että avainten kerääminen on hyödyllistä
3. Kompastele oven yli (onnea jälleen!)
4. Yhdistä koko sarja yksittäiseen palkkioon uloskäynnissä

Satunnaisessa etsinnässä mahdollisuus saada vahingossa loppuun tämä koko sarja pienenee eksponentiaalisesti jokaisen uuden vaaditun vaiheen myötä. Litteän DQN:n täytyy pohjimmiltaan saada onnea monta, monta kertaa, ennen kuin se näkee yhden positiivisen palkinnon, josta oppia.

---

## Hierarkkinen ratkaisu: hajota ja hallitse {#the-hierarchical-solution-divide-and-conquer}

Hierarkkinen RL jakaa pitkän tehtävän **kaksitasoiseksi rakenteeksi**:

| Taso | Kutsuttiin | Työ |
|-------|--------|-----|
| Korkea | **johtaja** | Valitsee seuraavan osatavoitteen |
| Matala  | **työntekijä** | Siirtyy kyseiseen osatavoitteeseen |

Juuri tällä tavalla ihmiset hoitavat monimutkaisia tehtäviä. Et suunnittele matkaasi käännökseltä ennen lähtöä. Sen sijaan:

- **Johtaja (sinä, kotona):** "Ensimmäinen pysäkki: huoltoasema. Seuraava pysäkki: moottoritien sisäänkäynti. Sitten: liittymä 42."
- **Työntekijä (sinä, ajo):** Käsittelee kaikki yksittäiset ohjauspäätökset jokaisen pysäkin saavuttamiseksi.

Johtaja ajattelee *tarkistuspisteissä*. Työntekijä ajattelee *ohjauspyörissä*.

---

## Miksi tämä voittaa tasaisen oppimisen pitkissä tehtävissä {#why-this-beats-flat-learning-on-long-tasks}

Työntekijän tarvitsee vain saavuttaa *seuraava osatavoite* – lyhyt tehtävä, jossa on selkeä, lähellä oleva palkkio. Se saa palautetta nopeasti ja oppii tehokkaasti.

Esimiehen tarvitsee vain päättää *alitavoitteiden järjestys* – paljon yksinkertaisempi ongelma kuin jokaisen yksittäisen askeleen suunnittelu.

Yhdessä nämä kaksi tasoa jakavat kovan pitkän horisontin ongelman kahdeksi helpoksi lyhyen horisontin ongelmaksi.

---

## Avain-oviverkkokoe {#the-key-door-grid-experiment}

Skriptimme testaa molempia lähestymistapoja **9x9 avoimessa ruudukossa** kahdella objektilla:

- **AVAIN** yhdessä kulmassa (kerää ensin).
- **DOOR** vastakkaisessa kulmassa (käyttää vain, jos sinulla on avain).

Ainoa todellinen palkinto on +1, kun agentti saapuu ovelle *avaimen noutamisen jälkeen. Tämä yksittäinen palkinto vaatii kaksi peräkkäistä osatehtävää, jotta se ketjutetaan oikein.

Kaksi agenttia kilpailee:

**Tasainen DQN:** Täytyy törmätä molempiin alitehtäviin oikeassa järjestyksessä vahingossa ja lähettää sitten signaali takaisin molempien kautta. Koska menestys vaatii kaksi onnellista löytöä yhdessä jaksossa, DQN oppii harvoin mitään hyödyllistä.

**Hierarkkinen edustaja:**
- Esimiehen sääntö: "Mene ensin avaimelle ja sitten ovelle."
- Työntekijä saa **+1 aina, kun hän saavuttaa alitavoitteen** - joko avain tai ovi.
- Kaksi erillistä lyhyttä tehtävää, joista jokaisella on selkeä lähellä oleva palkinto.

---

## Mitä kaaviot osoittavat {#what-the-charts-show}

![Pitkän horisontin tehtävien tulokset](outputs/long_horizon_tasks.png)

**Vasen – onnistumisprosentti ajan myötä:** Hierarkkinen agentti (sininen) oppii ratkaisemaan sokkelon paljon aikaisemmin kuin tasainen DQN (punainen). Litteä agentti saattaa lopulta myös oppia – jos jaksoja riittää – mutta hierarkkinen agentti pääsee perille nopeammin, koska sen oppimissignaali on tiheä ja paikallinen.

**Oikein – Lopullinen suorituskyky:** Pylväskaavio näyttää onnistumisprosentin keskiarvona viimeisen 500 jakson ajalta. Hierarkkisen agentin etu on selvä: ongelman jakaminen osatavoitteisiin tekee siitä selvitettävän.

---

## Missä pitkän horisontin ajattelu näkyy {#where-long-horizon-thinking-shows-up}

| Verkkotunnus | Esimerkki pitkästä horisontista |
|--------|---------------------|
| Robotiikka | Kokoa laite, jossa on 30 osaa järjestyksessä |
| Pelit | Voita shakkiottelu (useita liikkeitä, yksi voittaja) |
| Kieli | Kirjoita koko tutkimuspaperi (useita kirjoituspäätöksiä, yksi laatupiste) |
| Tiede | Suorita usean kuukauden kokeilu ja arvioi tulokset |

Juuri tästä syystä keksittiin Feudal Networks (arkkitehtuuri, jossa johtajat asettavat suunnattuja tavoitteita alemman tason työntekijöille) ja HIRO (hierarkkinen RL alitavoitteineen) - kun tasainen RL osui näihin ongelmiin, hierarkkisesta hajoamisesta tuli hallitseva strategia.

---

## Yhteys tavoitteen mukaisiin käytäntöihin {#the-connection-to-goal-conditioned-policies}

Huomaa, että hierarkkisessa agentissamme oleva **työntekijä** on pohjimmiltaan **tavoitteeseen perustuva käytäntö** — se vastaanottaa alitavoitteen ja siirtyy siihen. Tämä on HIRO:n ja siihen liittyvien papereiden vakiomuotoilu: johtaja asettaa tavoitteet, työntekijä on tavoitteellinen politiikka, joka jahtaa niitä.

Nämä kaksi ideaa – tavoitteellinen politiikka ja hierarkkinen rakenne – ovat siis saman kolikon kaksi puolta, minkä vuoksi ne esiintyvät yhdessä tässä moduulissa.

---

## Yhden lauseen yhteenveto {#one-sentence-summary}

> **Pitkän horisontin tehtävät ovat vaikeita, koska palkkio saapuu liian myöhään opettaakseen yksittäisiä päätöksiä – hierarkkinen RL ratkaisee tämän lisäämällä lähellä olevia alitavoitteita, jotka antavat työntekijän oppia nopeasti, kun johtaja käsittelee kokonaiskuvaa.**
