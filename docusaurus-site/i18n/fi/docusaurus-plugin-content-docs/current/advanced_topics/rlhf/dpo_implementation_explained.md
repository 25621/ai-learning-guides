# DPO: Ohita tuomari ja menee suoraan lähteeseen

## Suuri Idea

Klassisessa RLHF:ssä on kaksi vaihetta: ensin harjoittele palkintomalli ja käytä sitten PPO:ta
jahtaa sen tuloksia. DPO (Direct Preference Optimization) kysyy fiksua
kysymys:

*Jos palkintomalli on vain välivaihe, voimmeko ohittaa sen?*

Osoittautuu: kyllä. DPO kouluttaa kielimallin suoraan mieltymysten mukaan
parit, ilman erillistä tuomaria, ei PPO-näytteenottosilmukkaa eikä KL-kerrointa
virittää. Se käyttää yhtä tyylikästä kaavaa ja käyttäytyy kuin ohjattu oppiminen.

Tämä tekee DPO:sta yksinkertaisemman ajettavan, vakaamman ja nopeamman - minkä vuoksi se
on nopeasti tullut oletusvalinta monille avoimen lähdekoodin mukautetuille malleille.

## Analogia tosielämästä

Oletetaan, että valmennat opiskelijaa kirjoittamaan esseitä.

PPO-lähestymistapa on: palkkaa opettaja arvostelemaan esseitä ja pyydä sitten opiskelija
kirjoita essee esseen perään ja säädä opettajan arvosanojen perusteella.

DPO lähestymistapa on: näytä opiskelijalle kaksi essee kerrallaan ja sano:
"Tämä on parempi - nojaa tämän kaltaiseen kirjoittamiseen, pois siitä
yksi." Ei opettajaa keskellä. Opiskelija säätää suoraan
vertailuja.

Molemmat voivat toimia. DPO yleensä lopettaa nopeammin, koska kenenkään ei tarvitse kouluttaa ja
pitää erillinen opettaja.

## Kuinka oppiminen toimii (vain intuitio)

DPO käyttää samoja mieltymyspareja kuin palkkiomallinnus: kehote, valittu,
hylätty. Jokaiselle parille se esittää kaksi kysymystä:

1. Onko malli tullut **todennäköisemmin** tuottamaan valitun vastauksen?
   kuin vertailumalli olisi ollut?
2. Onko mallista tullut **vähemmän todennäköistä** tuottaa hylätty vastaus?
   kuin vertailumalli olisi ollut?

Harjoittelu työntää molemmat numerot oikeaan suuntaan kerralla. Ratkaisevaa,
vertailumalli on aina mukana vertailussa - se pelaa samaa
rooli KL-rangaistuksena PPO:ssa. Malli saa vaihtaa, mutta
siirtymät ovat aina *suhteessa* aloituspisteeseen.

Hienovarainen ja kaunis tulos DPO paperista on, että tämä yksittäinen menetys
-funktio vastaa matemaattisesti "kouluta palkkiomalli ja juokse sitten".
PPO KL-rangaistuksella." Sama kohde, yksinkertaisempi matka.

## Mitä Kokeilu osoittaa

Koulutimme käytännön suoraan 2 000 etusijaparille 300 aikakaudelle.

![DPO koulutus](outputs/dpo_implementation.png)

- **Vasen** - DPO-häviö laskee, kun malli oppii pitämään parempana valitusta
  hylättyjä vastauksia.
- **Keskitaso** - asetustarkkuus (kuinka usein käytäntö määrittää korkeamman
  implisiittinen palkkio valitusta vastauksesta) nousee noin 99 prosenttiin.
- **Oikein** - implisiittinen palkkiomarginaali kasvaa. DPO ei koskaan nimeä "palkintoa"
  mutta ero valittujen ja hylättyjen logaritmistodennäköisyyksien välillä, skaalattuna
  beta, voidaan lukea yhtenä. Se levenee tasaisesti, mikä tarkoittaa, että mallista tulee
  luottaa enemmän mieltymyksiinsä.

Huomaa, kuinka puhtaalta tämä näyttää verrattuna PPO:han. Näytteenottosilmukkaa ei ole, ei
tutkimuskohinaa, eikä erillistä palkkiomallia käynnissä. Jokainen aikakausi on a
puhdas valvottu-tyyppinen päivitys asetustietojoukon yli.

## Missä tämä sijaitsee RLHF-putkilinjassa

DPO on *vaihtoehto* perinteisen prosessin kahdelle vaiheelle:

- **Klassinen:** asetukset → palkkiomalli → PPO → tasattu malli.
- **DPO:** asetukset → tasattu malli. (Valmis.)

Saalista on, että DPO harjoittelee kiinteää asetustietosarjaa. PPO, koska
se poimii jokaisella kierroksella uusia vastauksia, ja se voi periaatteessa tutkia lisää.
Käytännössä DPO voittaa suurimman osan "kohdistaa kuratoituun asetustietosarjaan" -käytössä
tapauksia.

## Miksi tällä on väliä laboratorion ulkopuolella

"Ohita keskimmäinen mittaus" -kuvio on kaikkialla:

- Valmentaja korjaa uimarin muotoa esittelemällä mieluummin vierekkäin
  kuin pisteyttää jokaisen kierroksen sekuntikellolla.
- Valokuvaaja muokkaa kahta kuvaa kerralla ja valitsee paremman,
  "hyvän valokuvan" pisteytysrubriikin rakentamisen sijaan.
- Vuokrauspäällikkö vertailee kahta ansioluetteloa sen sijaan, että pisteyttäisi kutakin
  30 pisteen tarkistuslistaa vastaan.

Kun tarvitset vain *sijoituksen*, et tarvitse absoluuttista asteikkoa. DPO on
tämä näkemys soveltuu kielimalleihin.

## Yhden lauseen yhteenveto

**DPO muuttaa mieltymysparit suoraan paremmaksi malliksi ilman palkkiota
malli keskellä - yksinkertaisempi kuin PPO, ja usein yhtä hyvä.**
