# Palkkiomallinnus: Tietokoneen opettaminen sen, mitä ihmiset pitävät

## Suuri Idea

Palkkiomalli on pieni tuomari. Näytät sille kaksi vastausta samaan
Kysymys, kerro sille, kummasta henkilö piti enemmän, ja ajan myötä se oppii
antaa korkeampi pistemäärä vastauksille, jotka ihmiset haluaisivat.

Miksi me tarvitsemme sellaista tuomaria? Koska suurin osa siitä, mitä haluamme kieleltä
mallia on vaikea kirjoittaa matemaattiseksi kaavaksi. Ei ole olemassa yhtä yhtälöä
"avulias", "kohtelias" tai "hyvin kirjoitettu". Mutta ihmiset voivat melkein aina
osoita parempaa kahdesta vaihtoehdosta. Palkkiomalli tekee niistä yksinkertaisia
"tämä on parempi" äänestää arvosanaksi, jota oppimisalgoritmi voi käyttää.

## Analogia tosielämästä

Kuvittele, että opetat ystävääsi leipomaan brownieta.

Et anna heille 50-sivuista sääntökirjaa "mikä on hyvä brownie".
Sen sijaan maistat kahta erää ja sanot:

"Tämä on parempi."

Muutaman kierroksen jälkeen ystäväsi alkaa huomata kuvioita. Ehkä
tyhmempi voittaa aina. Ehkä ylipaistettu aina häviää. Sinun
ystävä rakentaa mentaalisen pisteytysjärjestelmän vertailuistasi.

Palkkiomalli tekee juuri tämän, mutta numeroilla. Sen ei tarvitse
tiedä *miksi* valittu vastaus on parempi. Se tarvitsee vain paljon "tätä lyöntiä
että" esimerkkejä ja se oppii vähitellen pistemäärän, joka on linjassa
mieltymykset.

## Kuinka oppiminen toimii (vain intuitio)

Jokainen esimerkki on kolmiosainen: kehote, **valittu** vastaus ja a
**hylätty** vastaus. Haluamme mallin antavan korkeamman pistemäärän
valitun kuin hylätyn - millään marginaalilla.

Harjoitustyö on hengeltään yksinkertainen:

- Valitun pistemäärä liian matala? Työnnä se ylös.
- Hylättyjen pistemäärä liian korkea? Paina se alas.
- Oletko jo oikeassa järjestyksessä selkeällä aukolla? Jätä heidät rauhaan.

Tätä nyökkäystä kutsutaan Bradley-Terry-tappioksi, ja se on vakioresepti
nykyaikaisissa RLHF-järjestelmissä.

## Mitä Kokeilu osoittaa

Koulutimme palkintomallin 2 000 synteettiselle mieltymysparille. Juoni
alla on kolme näkymää samasta harjoitusajosta.

![Palkkiomallikoulutus](outputs/reward_modeling.png)

- **Vasen** - tappio putoaa nopeasti. Malli on tulossa itsevarmemmaksi
  sen sijoituksista.
- **Keskitaso** - valintatarkkuus nousee lähes 100 prosenttiin. Melkein jokaisessa
  pari, valittu vastaus saa korkeamman pistemäärän kuin hylätty.
- **Oikea** - valittujen ja hylättyjen vastausten tulosjakaumat
  erillään. Alussa ne menivät päällekkäin; koulutuksen jälkeen valitut vastaukset
  istu selkeästi oikealle.

Tuo ero on koko asian ydin. Myöhempi vaihe (PPO tai DPO) voidaan nyt käyttää
tämä pistemäärä optimointikohteena.

## Missä tämä sijaitsee RLHF-putkilinjassa

Etenemissuunnitelmassa RLHF kuvataan "mallien mukauttamiseksi ihmisten mieltymyksiin".
Palkkiomalli on vaihe yksi kolmesta:

1. **Palkintamalli (tämä tiedosto)** - muuntaa etusijaäänet pisteiksi.
2. **PPO-hienosäätö** - työnnä kielimallia kohti korkeampia pisteitä
   pysyen samalla lähellä alkuperäistä käyttäytymistään.
3. **DPO** – uudempi pikakuvake, joka ohittaa palkkiomallin kokonaan.

Joten palkitsemismallinnus on silta *ihmisarvion* ja
*koneoptimointi*. Tee tämä silta väärin ja jokainen alavirran vaihe
ohjataan pois kurssilta.

## Miksi tällä on väliä laboratorion ulkopuolella

Sama ajatus näkyy monessa paikassa:

- **Suositusjärjestelmät** oppivat mitä pidät napsautuksista, ohituksista ja
  katseluun käytetty aika.
- **Hakukoneet** oppivat sijoituksen "mitä tulosta napsautit".
- **Ravintolat** oppivat suosittuja ruokia toistuvista tilauksista, eivät tilauksista
  asiakkaat kirjoittavat esseitä siitä, mistä he pitivät.

Aina kun on helpompi *vertaa* kuin *arvostella*, palkitsemismalli on se
oikea työkalu.

## Yhden lauseen yhteenveto

**Palkintamalli on oppinut tuomari, joka päättää "tämä on parempi"
muu RLHF voi optimoida mieltymykset numeerisiksi pisteiksi.**
