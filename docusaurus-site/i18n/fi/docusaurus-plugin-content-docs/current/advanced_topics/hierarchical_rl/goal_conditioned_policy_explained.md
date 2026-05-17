# Tavoitteen mukainen käytäntö

## Suuri idea: yksi politiikka hallitsee niitä kaikkia {#the-big-idea-one-policy-to-rule-them-all}

Kuvittele, että olet jakelukuljettaja. Et tarvitse täysin erilaista taitosarjaa jokaiselle osoitteelle. Osaat ajaa, lukea karttaa ja navigoida liikenteessä – liität vain *tämän päivän määränpään* ja lähdet.

**Tavoitteeseen perustuva politiikka** toimii samalla tavalla. Sen sijaan, että kouluttaisimme yhtä agenttia, joka voi saavuttaa vain yhden kiinteän tavoitteen, koulutamme yhden agentin, joka hyväksyy minkä tahansa tavoitteen syötteenä ja selvittää, miten siihen päästään.

---

## Miten se eroaa tavallisesta RL:stä {#how-it-differs-from-standard-rl}

Normaalissa RL:ssä (kuten opetussuunnitelman aikaisemmissa vaiheissa käsiteltiin) palkitsemistoiminto leivotaan seuraavasti: "tavoita solu (7, 7), saat +1." Agentti oppii täsmälleen yhden asian: kuinka tavoittaa *tuo* solu.

Tavoitteellisessa RL:ssä palkkio riippuu siitä, saavuttaako agentti *mikä tahansa tavoite, joka sille tällä kertaa annettiin*. Politiikka oppii:

> **"Mitä minun pitäisi tehdä, kun otetaan huomioon missä olen ja missä haluan olla?"**

Tavoite kulkee *agentin mukana*, kuten navigointisovellukseen kirjoitettu määränpää.

---

## Harva palkitsemisongelma {#the-sparse-reward-problem}

Tässä on saalis: harvoista palkinnoista oppiminen (vain +1 maalissa, 0 kaikkialla muualla) on julman vaikeaa. Useimmat yritykset epäonnistuvat – agentti vaeltelee satunnaisesti, ei koskaan törmää maaliin, eikä verkko saa mitään hyödyllistä opittavaa.

Kuvittele, että yrität oppia heittämään tikkaa sidottuna. Heität tuhat kertaa ja menetät aina. Tuhansien epäonnistumisten jälkeen et vieläkään tiedä, miltä "hyvä heitto" tuntuu.

Tässä tulee esiin **Hindsight Experience Replay (HER)**.

---

## Jälkikatselukokemuksen uusinta: epäonnistuminen eteenpäin {#hindsight-experience-replay-failing-forward}

HÄNEN temppu on ihanan yksinkertainen. Epäonnistuneen jakson jälkeen HÄN kysyy:

> *"Vaikka et saavuttanutkaan tavoitettasi… mihin oikein päädyit?"*

Sitten se **toistaa saman jakson**, mutta teeskentelee agentin todellista lopullista sijaintia **olleen** tavoite koko ajan. Yhtäkkiä epäonnistuneesta jaksosta tulee onnistunut - eri tavoitteen saavuttamiseksi.

Se on kuin epäonnistunut koripalloilija, joka ampuu jatkuvasti vanteeseen ja puuttuu. HÄN sanoisi: "Okei, osuit vasempaan seinään joka kerta. Onnittelut – olet loistava lyömään vasenta seinää! Kirjataan ne heitot onnistuneiksi vasen seinän lyöntiyrityksiksi." Ajan myötä pelaaja kehittää taitojaan lyödä *mitä tahansa* maalia ja siirtää sen lopulta oikeaan vanteeseen.

Tämä muuttaa tuhansia "epäonnistumisia" rikkaaksi kirjastoksi *onnistuneita* navigointeja moniin eri paikkoihin. Agentti oppii saavuttamaan ne kaikki, mikä yleistyy todelliseen kohteeseen.

---

## Analogia tosielämästä: taapero oppii pinoamaan lohkoja {#the-real-life-analogy-toddler-learning-to-stack-blocks}

Taaperolapsi, joka yrittää laittaa lohkon ämpäriin, missaa jatkuvasti. Mutta jokainen "miss" laskeutuu korttelin *johonkin*. Jos toistat jokaisen poissaolon sanomalla "yritit laittaa sen *oikealle* – ja teit sen!", taapero kehittää hienomotoriikkaa koko pöydässä. Pian he voivat asettaa lohkon mihin tahansa - myös ämpäriin.

---

## Mitä koodimme tekee {#what-our-code-does}

Käsikirjoitus `goal_conditioned_policy.py` kulkee **7x7 sokkelossa** seinillä. Jokaisen jakson alussa valitaan satunnainen maalisolu. Agentin on löydettävä se.

Käytännössä on kaksi syötettä jokaisessa vaiheessa:
1. Missä edustaja tällä hetkellä on
2. Minne se haluaa mennä

Jokaisen jakson (onnistuneen tai epäonnistuneen) jälkeen HER luo useita synteettisiä "onnistuksia" merkitsemällä todelliset vieraillut paikat uudelleen vaihtoehtoisiksi tavoitteiksi.

Koulutus kestää 3 000 jaksoa hidastuvalla tutkimusnopeudella – agentti tutkii aluksi enemmän ja luottaa sitten yhä enemmän oppimaansa.

---

## Mitä kaaviot osoittavat {#what-the-charts-show}

![Tavoitteen mukaiset käytäntötulokset](outputs/goal_conditioned_policy.png)

**Vasemmalla – onnistumisprosentti harjoittelusta:** Jokainen jakso on joko menestys (tavoitteen saavuttaminen) tai epäonnistuminen. Käyrä nousee tasaisesti agentin yleisen navigointitaidon kehittyessä. Lopulta agentti saavuttaa minkä tahansa tavoitteen melkein joka kerta.

**Oikein — maalin onnistumisprosentin lämpökartta:** Harjoittelun jälkeen testaamme agenttia kaikissa mahdollisissa maalisoluissa ja värjäämme jokaisen solun sen mukaan, kuinka usein agentti saavuttaa sen. Vihreä tarkoittaa, että agentti saavuttaa luotettavasti kyseisen kohdan; punainen tarkoittaa, että se kamppailee edelleen. Hyvin koulutettu agentti näyttää enimmäkseen vihreää koko sokkelossa.

---

## Missä tämä näkyy todellisessa maailmassa {#where-this-shows-up-in-the-real-world}

| Sovellus | "tavoite" |
|-------------|------------|
| Robotin käsi ulottuu | Kohdista 3D-asento |
| Itse ajava auto | GPS-koordinaatti |
| Kielimallin avustaja | Käyttäjän ohje |
| Videopeli, ei-pelaajahahmo | Mikä tahansa reittipiste kartalla |

Tavoiteehtoiset käytännöt ovat yksi HIRO:n (Hierarchical RL with subgoals) rakennuspalikoista – korkean tason johtaja valitsee osatavoitteen, ja matalan tason työntekijä on juuri tällainen tavoitteellinen politiikka.

---

## Yhden lauseen yhteenveto {#one-sentence-summary}

> **Tavoitteeseen ehdollinen käytäntö on agentti, joka voi navigoida mihin tahansa kohteeseen – ja HER mahdollistaa epäonnistumisesta oppimisen teeskentelemällä, että jokainen menetetty laukaus on suunnattu minne tahansa se laskeutuikin.**
