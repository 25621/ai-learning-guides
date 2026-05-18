# Lineaarinen Q-Learning CartPolelle 🎪

## Mikä on CartPole?

Kuvittele luudanvarsi, joka on tasapainossa sormellasi. Jos liikutat sormeasi vasemmalle tai oikealle
vain vähän, voit estää luudan varren putoamisen. Se on **CartPole**!

Pieni robotti istuu kärryissä (pyörillä oleva laatikko) ja sen päällä on sauva.
Robotti voi työntää kärryä vain **vasemmalle** tai **oikealle**. Sen on opittava pitämään
tanko tasapainossa mahdollisimman pitkään – aivan kuten tasapainotat luudanvartta!

Robotti voi nähdä 4 asiaa maailmasta:
1. Missä kärry on
2. Kuinka nopeasti kärry liikkuu
3. Kuinka paljon pylväs kallistuu
4. Kuinka nopeasti sauva kallistuu

---

## Suuri ongelma: Liian monta tilaa!

Muistatko Q-learningin vaiheesta 2? Se käytti suurta taulukkoa muistaakseen, kuinka hyvä kukin toiminto
oli kussakin tilanteessa (tilassa). Se toimi hyvin Frozen Lakelle – siellä oli vain 16
ruutua jäällä.

Mutta CartPole on erilainen! Kärry voi olla **missä tahansa asennossa**, liikkua **millä tahansa nopeudella**,
tangon ollessa **missä tahansa kulmassa**. Pohjimmiltaan on olemassa **ääretön määrä mahdollisia tiloja**! Emme voi
tee taulukko, jossa on äärettömät rivit. Tarvitsemme universumin kokoisen muistikirjan!

**Esimerkki tosielämästä:** Kuvittele, että opettelet ajamaan pyörällä. Jokaista mahdollista huojuntaa ei voi muistaa
– niitä on aivan liikaa! Sen sijaan opit **säännön**: "Kun nojaan vasemmalle,
käännän oikealle; kun nojaan oikealle, käännän vasemmalle." Yksinkertainen sääntö pätee KAIKKIIN heilahteluihin.

---

## Ratkaisu: Maaginen kaava

**Lineaarinen funktion likiarvo** korvaa jättimäisen taulukon **pienellä kaavalla**:

> **Pistemäärä (tilanne, toiminto) = w₁ × kärryn_sijainti + w₂ × kärryn_nopeus + w₃ × tangon_kulma + w₄ × tangon_kulmanopeus**

- `w`-lukuja kutsutaan **painoiksi** – ne ovat kuin nuppeja, joita voi kiertää
- Opimme **eri painot jokaiselle toiminnolle** (työnnä vasemmalle ja työnnä oikealle)
- Kaava antaa pistemäärän siitä, kuinka hyvä kukin toiminta on juuri nyt

**Tosielämän esimerkki:** Ajattele yksinkertaista reseptiä: "1 kuppi jauhoja + 2 munaa + ½ kuppi voita."
Painot (1, 2, ½) kertovat, kuinka paljon kullakin ainesosalla on merkitystä. Me opimme
resepti hyviin päätöksiin!

---

## Miten se oppii?

Robotti kokeilee asioita, saa palautetta ja säätää painoja:

1. **Robotti työntää kärryä** (valitsee toiminnon, jolla on korkein pistemäärä)
2. **Fysiikka tapahtuu** (tanko kallistuu hieman, kärry liikkuu)
3. **Robotti saa palkinnon** (+1 jokaisesta askeleesta, jossa sauva pysyy pystyssä, 0, jos se putoaa)
4. **Robotti kysyy:** "Oliko todellinen tulos parempi vai huonompi kuin ennustin?"
5. **Robotti säätää painoja** ollakseen ensi kerralla lähempänä todellisuutta

Tämä on **Semi-Gradient TD -päivitys** - fantastinen nimi sanalle "nukista reseptiä hieman
yllätyksen perusteella."

> **Uusi paino = vanha paino + oppimisnopeus × (mitä todella tapahtui − mitä ennustin) × Ominaisuus**

---

## Mitä koodimme löysi

Kun juokset `linear_q_cartpole.py`, robotti:

- Alkaa kauheasti (sauva putoaa 10-30 askelta)
- Oppii vähitellen hyvät painot yli 3000 yrittämällä
- Pitää lopulta tangon tasapainossa 100–400+ askelta!

Kaaviossa näkyy **oppimiskäyrä** — kuinka pisteet paranevat ajan myötä.
Se tulee olemaan kuoppainen (oppiminen ei ole koskaan sujuvaa!), mutta trendin pitäisi nousta.

---

## Miksi tämä on siistiä (ja rajoitettua!)

**Hienoa:** Pieni kaava, jossa on vain 8 numeroa (4 painoa × 2 toimintoa), voi pitää tangon tasapainossa!
Mitään jättimäistä taulukkoa ei tarvita.

**Rajoitettu:** Kaava on liian yksinkertainen monimutkaisiin tehtäviin. Se olettaa aina, että suuremmat luvut
tarkoittavat suurempia vaikutuksia (mikä ei aina pidä paikkaansa). Monimutkaisempia pelejä, kuten Ataria, varten tarvitsemme
**neuroverkkoja** – ja juuri sitä DQN tekee!

---

## Avainsanasto

| sana | Merkitys |
|------|---------|
| **Ominaisuus** | Yksi mitattavissa oleva asia maailmassa (esim. tangon kulma) |
| **Paino** | Kuinka paljon ominaisuus vaikuttaa päätökseen |
| **Lineaarinen** | Kaava on vain kerto- ja yhteenlasku (ei monimutkaisia käyriä) |
| **Puoligradientti** | Päivittää painot seuraten virheen vähenemistä |
| **Funktioapproksimaatio** | Kaavan käyttäminen taulukon sijaan |

---

## Mitä seuraavaksi?

Lineaarinen approksimaatio on kuin käyttäisit suoraa viivainta käyrän piirtämiseen – se toimii hyvin
yksinkertaisia muotoja, mutta ei monimutkaisia. Atari-peleissä, joissa on miljoonia mahdollisia tilanteita,
tarvitsemme **Deep Q-Networks (DQN)** - hermoverkkoja, jotka voivat oppia paljon monimutkaisempia
kuvioita. Se on seuraavassa tiedostossa!
