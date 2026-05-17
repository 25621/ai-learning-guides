# PPO-Feintuning: Ein Modell polieren, ohne es zu zerstören

## Die Kernidee

Sobald wir ein Belohnungsmodell (Reward Model) haben, das Antworten bewertet, möchten wir, dass unser Sprachmodell Antworten mit höheren Scores produziert. PPO (Proximal Policy Optimization) erledigt genau das – aber es fügt einen Sicherheitsgurt hinzu, damit das Modell nicht nur den Score jagt und dabei vergisst, wie man normalen Text schreibt.

Betrachten Sie es als einen Polierschritt. Das Modell spricht bereits flüssig; wir geben ihm nur einen Schubs, damit es so spricht, wie es das Belohnungsmodell belohnt, während seine Stimme erkennbar bleibt.

## Eine Analogie aus dem echten Leben

Stellen Sie sich einen Koch vor, der bereits gut kocht, aber nun lernt, einen ganz bestimmten Restaurantkritiker zufrieden zu stellen.

Nach jedem Gericht gibt der Kritiker eine Note ab. Auf den Koch wirken zwei Kräfte:

1. **Einen höheren Score erzielen.** So kochen, wie der Kritiker es mag.
2. **Nicht unkenntlich werden.** Wenn der Koch seinen eigenen Stil komplett aufgibt – und Salz literweise hineinschüttet, nur um einen Score zu jagen –, wird das Essen seltsam. Die Kunden bleiben aus.

PPO erfasst beide Kräfte:

- Der **Belohnungs-Teil** drückt das Modell in Richtung der Antworten, die der Richter mag.
- Der **KL-Strafterm** zieht das Modell zurück zu der Art und Weise, wie es vor Beginn des Trainings gesprochen hat. KL ist einfach eine Methode, um zu messen, „wie sehr sich das neue Verhalten vom alten unterscheidet“.

Zusammen sagen sie: *Werde besser, aber bleib du selbst.*

## Wie das Lernen funktioniert (Intuition)

Jede Trainingsrunde sieht so aus:

1. Nimm einige Prompts. Lass das aktuelle Modell Antworten erzeugen.
2. Bewerte die Antworten mit dem Belohnungsmodell.
3. Vergleiche mit dem **Referenzmodell** – einer eingefrorenen Kopie des Modells vor dem Training. Wenn die neuen Antworten extrem abweichen, ziehe eine KL-Strafe von der Belohnung ab.
4. Bewege das Modell in Richtung der Antworten, die gut abgeschnitten haben.

Das „Proximal“ in PPO bedeutet: *Mache keine großen Sprünge*. Jedes Update ist ein kleiner, vorsichtiger Schritt. Große Sprünge beim Training von Strategien (Policies) verursachen Abstürze, weshalb frühere Methoden wie der einfache Policy-Gradient so instabil waren.

## Was das Experiment zeigt

Wir beginnen mit einer frischen Policy und einem trainierten Belohnungsmodell. PPO läuft über 150 Iterationen, wobei Batches von Antworten gezogen und die Policy aktualisiert wird.

![PPO-Training](outputs/ppo_fine_tuning.png)

- **Links** — der durchschnittliche Score des Belohnungsmodells steigt stetig an. Die Policy lernt, Antworten zu produzieren, die der Richter mag.
- **Mitte** — die KL-Divergenz zum Referenzmodell wächst. Die Policy entfernt sich von ihrem Ausgangspunkt. Das ist zu erwarten, aber wenn sie unkontrolliert wachsen würde, würde das Modell in Unsinn abgleiten.
- **Rechts** — die geformte Belohnung (rohe Belohnung minus KL-Strafe) folgt anfangs eng der rohen Belohnung, fällt dann aber zurück, wenn KL steigt. Die Strafe tut ihren Job: Sie lässt das Modell dafür „bezahlen“, wenn es zu weit abdriftet.

In einem echten RLHF-System stellt man den KL-Koeffizienten so ein, dass der Score immer noch steigt, das Modell aber kohärent bleibt. Eine zu kleine Strafe führt dazu, dass das Modell die Belohnung „hackt“, indem es seltsame, sich wiederholende Phrasen ausgibt. Eine zu große Strafe verhindert jegliche Verbesserung des Modells.

## Wo PPO in der RLHF-Pipeline steht

Dies ist Schritt zwei des klassischen RLHF-Rezepts:

1. Trainiere ein Belohnungsmodell aus Präferenzen.
2. **Führe ein Feintuning des Sprachmodells mit PPO unter Verwendung dieses Belohnungsmodells durch.**
3. (Optional) Überspringe Schritt 2 mit DPO, wenn Sie einen einfacheren Weg bevorzugen.

PPO ist das Arbeitstier, das Unternehmen wie OpenAI und Anthropic für die erste Welle ausgerichteter Modelle verwendeten, einschließlich InstructGPT und dem ursprünglichen ChatGPT.

## Warum das außerhalb des Labors wichtig ist

Das Muster „Verbessern, aber nicht abdriften“ findet sich überall:

- Ein Pianist, der ein schwieriges Stück übt, ändert nicht seine gesamte Technik, um eine Passage zu meistern – das würde den Rest des Konzerts ruinieren.
- Ein Unternehmen, das eine Website optimiert, um die Anmeldungen zu erhöhen, muss die Marke für bestehende Nutzer immer noch erkennbar halten.
- Eine Fabrik, die einen Regler in einem Prozess anpasst, hält die anderen nah an den bewährten Einstellungen.

PPO ist einfach eine vorsichtige, mathematisch formulierte Version dieser universellen Idee.

## Zusammenfassung in einem Satz

**PPO-Feintuning bewegt ein Modell in Richtung einer höheren Belohnung, während eine KL-Strafe es nah an seinem ursprünglichen Verhalten hält – verbessere dich, aber bleib dir treu.**
