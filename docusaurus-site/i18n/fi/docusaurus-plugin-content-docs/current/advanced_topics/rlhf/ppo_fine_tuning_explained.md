# PPO-hienosäätö: mallin kiillotus rikkomatta sitä

## Suuri Idea

Kun meillä on palkitsemismalli, joka pisteyttää vastauksia, haluamme kielemme
malli tuottaa korkeampia pisteitä saavia vastauksia. PPO (proksimaalinen käytäntö
Optimointi) tekee juuri tämän - mutta se lisää turvavyön niin malliin
ei jahtaa pisteet ja unohtaa kuinka kirjoittaa normaalia tekstiä.

Ajattele sitä kiillotusaskelena. Malli puhuu jo sujuvasti; me vain
työnnä se puhumaan tavalla, jonka palkkiomalli palkitsee, pitäen samalla sen
ääni tunnistettavissa.

## Analogia tosielämästä

Kuvittele kokki, joka valmistaa jo hyvin, mutta nyt oppii miellyttämään a
erityinen ruokakriitikko.

Jokaisen aterian jälkeen kriitikko antaa pisteytyksen. Kokilla on kaksi painetta:

1. **Saat korkeammat pisteet.** Valmista kriitikon pitämällä tavalla.
2. **Älä tule tunnistamattomaksi.** Jos kokki hylkää oman tyylinsä
   kokonaan - suolan heittäminen kupin luokse vain jahdatakseen pisteet -
   ruoasta tulee outoa. Asiakkaat lakkaavat tulemasta.

PPO kaappaa molemmat paineet:

- **palkkio**-osa työntää mallia kohti vastauksia, joista tuomari pitää.
- **KL-rangaistus** -osa vetää mallia takaisin siihen, miten se puhui ennen
  koulutus alkoi. KL on vain tapa mitata "kuinka erilainen on
  uutta käytöstä vanhasta käytöksestä."

Yhdessä he sanovat: *parane, mutta pysy omana itsenäsi*.

## Kuinka oppiminen toimii (vain intuitio)

Jokainen harjoituskierros näyttää tältä:

1. Ota vastaan kehotteita. Anna nykyisen mallin tuottaa vastauksia.
2. Pistele vastaukset palkintomallilla.
3. Vertaa **referenssimalliin** - mallin jäädytettyyn kopioon
   ennen harjoittelua. Jos uudet vastaukset ovat villisti erilaisia,
   vähennä palkkiosta KL-rangaistus.
4. Työnnä mallia kohti vastauksia, jotka saivat hyvän tuloksen.

"Proksimaalinen" PPO:ssa tarkoittaa *älä ota suuria hyppyjä*. Jokainen päivitys on a
pieni, varovainen askel. Suuret hypyt käytäntökoulutuksessa aiheuttavat kaatumisia, mikä on
miksi aikaisemmat menetelmät, kuten vaniljakäytännön gradientti, olivat niin epävakaita.

## Mitä Kokeilu osoittaa

Aloitamme uudella käytännöllä ja koulutetulla palkitsemismallilla. PPO suorittaa 150
iteraatiot, vastauserien näytteenotto ja käytännön päivittäminen.

![PPO koulutus](outputs/ppo_fine_tuning.png)

- **Vasen** - palkintomallin keskimääräinen pistemäärä nousee tasaisesti. Käytäntö on
  oppia tuottamaan vastauksia, joista tuomari pitää.
- **Keski** - KL:n ero vertailumallista kasvaa. Käytäntö on
  siirtyä pois siitä, mistä se alkoi. Tätä odotetaan, mutta jos se kasvaa
  tarkistamatta malli ajautuisi hölynpölyyn.
- **Oikein** - muotoiltu palkinto (raaka palkinto miinus KL-rangaistus) kappaleita
  raaka palkkio ensin tiukasti, sitten jää jälkeen KL:n noustessa. The
  rangaistus tekee tehtävänsä: saa mallin "maksamaan" liian pitkälle ajautumisesta.

Oikeassa RLHF-järjestelmässä virität KL-kerrointa, kunnes pisteet ovat edelleen
nousee, mutta malli pysyy yhtenäisenä. Liian pieni rangaistus ja malli
hakkeroi palkinnon lähettämällä outoja toistuvia lauseita. Liian suuri ja
malli ei koskaan parane.

## Missä tämä sijaitsee RLHF-putkilinjassa

Tämä on klassisen RLHF-reseptin toinen vaihe:

1. Kouluta palkintomalli mieltymysten perusteella.
2. **Hienoviritä kielimalli PPO:lla käyttämällä tätä palkintomallia.**
3. (Valinnainen) Ohita vaihe 2 DPO:lla, jos haluat yksinkertaisemman polun.

PPO on työhevonen, jota yritykset, kuten OpenAI ja Anthropic, käyttivät
kohdistettujen mallien ensimmäinen aalto, mukaan lukien InstructGPT ja alkuperäinen
ChatGPT.

## Miksi tällä on väliä laboratorion ulkopuolella

"Paranna, mutta älä ajaudu" -kuvio näkyy kaikkialla:

- Kovaa kappaletta harjoitteleva pianisti ei muuta koko tekniikkaansa
  naulaa yksi kohta - se rikkoisi kappaleen loput.
- Yrityksen, joka säätelee verkkosivustoa lisätäkseen ilmoittautumisia, on silti säilytettävä brändi
  nykyisten käyttäjien tunnistettavissa.
- Tehdassäätö yksi nuppi prosessissa pitää muut lähellä
  tunnetut hyvät asetukset.

PPO on vain varovainen versio tästä universaalista ideasta, joka on kirjoitettu matematiikassa.

## Yhden lauseen yhteenveto

**PPO-hienosäätö työntää mallia kohti korkeampaa palkkiota KL-rangaistuksen ohella
pitää sen lähellä alkuperäistä käyttäytymistään - paranna, mutta pysy omana itsenäsi.**
