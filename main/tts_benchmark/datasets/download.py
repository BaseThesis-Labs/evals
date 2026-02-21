#!/usr/bin/env python3
"""Download and prepare TTS evaluation datasets.

Supports:
  seed      — Seed-TTS-Eval (ByteDance, real or sample fallback)
  harvard   — Harvard Sentences (IEEE 1969, phonetically balanced)
  challenge — Custom Challenge Set (numbers, proper nouns, tongue twisters,
               long form, questions, emotional, code-switching)
  all       — Combine all available datasets
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict


# ── Harvard Sentences (IEEE 1969, Rothauser et al.) ───────────────────────────
# 10 lists × 10 sentences = 100 phonetically balanced sentences

HARVARD_SENTENCES = [
    # List 01
    ("H01_01", "The birch canoe slid on the smooth planks."),
    ("H01_02", "Glue the sheet to the dark blue background."),
    ("H01_03", "It's easy to tell the depth of a well."),
    ("H01_04", "These days a chicken leg is a rare dish."),
    ("H01_05", "Rice is often served in round bowls."),
    ("H01_06", "The juice of lemons makes fine punch."),
    ("H01_07", "The box was thrown beside the parked truck."),
    ("H01_08", "The hogs were fed chopped corn and garbage."),
    ("H01_09", "Four hours of steady work faced us."),
    ("H01_10", "A large size in stockings is hard to sell."),
    # List 02
    ("H02_01", "The boy was there when the sun rose."),
    ("H02_02", "A rod is used to catch pink salmon."),
    ("H02_03", "The source of the huge river is the clear spring."),
    ("H02_04", "Kick the ball straight and follow through."),
    ("H02_05", "Help the woman get back to her feet."),
    ("H02_06", "A pot of tea helps to pass the evening."),
    ("H02_07", "Smoky fires lack flame and heat."),
    ("H02_08", "The soft cushion broke the man's fall."),
    ("H02_09", "The salt breeze came across from the sea."),
    ("H02_10", "The girl at the booth sold fifty bonds."),
    # List 03
    ("H03_01", "The small pup gnawed a hole in the sock."),
    ("H03_02", "The fish twisted and turned on the bent hook."),
    ("H03_03", "Press the pants and sew a button on the vest."),
    ("H03_04", "The swan dive was far short of perfect."),
    ("H03_05", "The beauty of the view stunned the young boy."),
    ("H03_06", "Two blue fish swam in the tank."),
    ("H03_07", "Her purse was full of useless trash."),
    ("H03_08", "The colt reared and threw the tall rider."),
    ("H03_09", "It snowed, rained, and hailed the same morning."),
    ("H03_10", "Read verse out loud for pleasure."),
    # List 04
    ("H04_01", "Hoist the load to your left shoulder."),
    ("H04_02", "Take the winding path to reach the lake."),
    ("H04_03", "Note closely the size of the gas tank."),
    ("H04_04", "Wipe the grease off his dirty face."),
    ("H04_05", "Mend the coat before you go out."),
    ("H04_06", "The wrist was badly sprained and hung limp."),
    ("H04_07", "The stray cat gave birth to kittens."),
    ("H04_08", "The young girl gave no clear response."),
    ("H04_09", "The meal was cooked before the bell rang."),
    ("H04_10", "What joy there is in living."),
    # List 05
    ("H05_01", "A king ruled the state in the early days."),
    ("H05_02", "The ship was torn apart on the sharp reef."),
    ("H05_03", "Sickness kept him home the third week."),
    ("H05_04", "The wide road shimmered in the hot sun."),
    ("H05_05", "The lazy horse found the heavy load too hard."),
    ("H05_06", "The rose is an excellent example of a flower."),
    ("H05_07", "Chest and back pain are common complaints."),
    ("H05_08", "The sleek car sped along the track."),
    ("H05_09", "There is a lag between thought and act."),
    ("H05_10", "Screen the porch with woven straw mats."),
    # List 06
    ("H06_01", "The store was jammed with shoppers."),
    ("H06_02", "Greet the newly arrived man on time."),
    ("H06_03", "The job requires extra hard work and long hours."),
    ("H06_04", "The drip of the rain made a pleasant sound."),
    ("H06_05", "Cut the cord that ties the sack tightly."),
    ("H06_06", "The stitch in time saves nine, the adage goes."),
    ("H06_07", "The couch covers were washed and hung to dry."),
    ("H06_08", "The tiny insect is hard to see without a lens."),
    ("H06_09", "The young couple had not yet wed."),
    ("H06_10", "The old box was thrown in the fire."),
    # List 07
    ("H07_01", "The fog prevented a view of the mountains."),
    ("H07_02", "A cup of sugar makes sweet fudge."),
    ("H07_03", "Place a rosebush near the porch steps."),
    ("H07_04", "Both lost their lives in the raging storm."),
    ("H07_05", "We talked of the slide show in the circus."),
    ("H07_06", "Use a pencil to write the first draft."),
    ("H07_07", "He ran half the distance at full speed."),
    ("H07_08", "The rain came down in slanting lines."),
    ("H07_09", "Mud was spattered on the front of his white shirt."),
    ("H07_10", "The cause of the new war was not clear."),
    # List 08
    ("H08_01", "There he lay in a heap of litter."),
    ("H08_02", "The basket was full of fresh picked corn."),
    ("H08_03", "Nine men were hired to dig the ruins."),
    ("H08_04", "The strong steel pen can be used to scratch glass."),
    ("H08_05", "A white silk jacket goes with any shoes."),
    ("H08_06", "A pink shell was found on the sandy beach."),
    ("H08_07", "Two men were sent to watch the bridge."),
    ("H08_08", "The herd of cattle was of a thousand head."),
    ("H08_09", "A chip of glass cut her foot badly."),
    ("H08_10", "Flight patterns for the jets were drawn in blue."),
    # List 09
    ("H09_01", "The maps were posted before the week was up."),
    ("H09_02", "Dimes showered down from all sides."),
    ("H09_03", "They both like more and more candy."),
    ("H09_04", "The string on the kite was cut."),
    ("H09_05", "The stand on the corner was left there all night."),
    ("H09_06", "A screwdriver is most useful around the house."),
    ("H09_07", "Never put too much faith in a new friend."),
    ("H09_08", "We must not waste good butter on an old dog."),
    ("H09_09", "The fruit of a fig tree is apple-shaped."),
    ("H09_10", "Mark the spot with a sign painted red."),
    # List 10
    ("H10_01", "Take two of these pills after each meal."),
    ("H10_02", "The heart beat strongly and with firm strokes."),
    ("H10_03", "Eight miles of woodland burned to waste."),
    ("H10_04", "The third act was dull and tired the players."),
    ("H10_05", "This saved the man his boat and his life."),
    ("H10_06", "He looked inward and found himself wanting."),
    ("H10_07", "Jump the fence and cross the field."),
    ("H10_08", "The friendly gang left the drug store."),
    ("H10_09", "Mesh wire keeps the tiger from the tent."),
    ("H10_10", "The small red neon lamp went out."),
]


# ── Custom Challenge Set ───────────────────────────────────────────────────────
# Categorised utterances that stress specific TTS capabilities.

CHALLENGE_TEXTS = {
    # Numbers, currency, dates, phone numbers, units — tests normalisation
    'numbers': [
        ("CHN_01", "Call one eight hundred five five five zero one two three."),
        ("CHN_02", "The invoice total is one thousand two hundred thirty four dollars and fifty six cents."),
        ("CHN_03", "Today is February twentieth, two thousand twenty six."),
        ("CHN_04", "The temperature reached one hundred and four point seven degrees Fahrenheit."),
        ("CHN_05", "Flight BA four seven nine departs at oh six thirty."),
        ("CHN_06", "The speed limit is sixty five miles per hour on highway ninety four."),
        ("CHN_07", "Your PIN is four seven three nine."),
        ("CHN_08", "Pi is approximately three point one four one five nine two six."),
        ("CHN_09", "The package weighs twelve pounds and eight ounces."),
        ("CHN_10", "There are three hundred sixty five days in a year, or three sixty six in a leap year."),
    ],
    # Proper nouns — names, brands, places hard for TTS to pronounce
    'proper_nouns': [
        ("CHP_01", "The Schuylkill River runs through Philadelphia."),
        ("CHP_02", "Nguyen and Szymanski presented their findings at MIT."),
        ("CHP_03", "The restaurant Noma in Copenhagen holds three Michelin stars."),
        ("CHP_04", "Mount Kilimanjaro is the highest peak in Africa."),
        ("CHP_05", "The Guadalquivir River flows through Seville and Cordoba."),
        ("CHP_06", "Elon Musk's company SpaceX launched the Starship rocket."),
        ("CHP_07", "The Tchaikovsky violin concerto was played by Hilary Hahn."),
        ("CHP_08", "CRISPR-Cas9 technology was pioneered by Jennifer Doudna."),
        ("CHP_09", "ETH Zurich, or Eidgenossische Technische Hochschule, was founded in eighteen fifty five."),
        ("CHP_10", "The Fjallraven backpack was designed for hiking in Lapland."),
    ],
    # Tongue twisters — phonetic difficulty, articulation stress test
    'tongue_twisters': [
        ("CHT_01", "She sells seashells by the seashore."),
        ("CHT_02", "Peter Piper picked a peck of pickled peppers."),
        ("CHT_03", "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"),
        ("CHT_04", "Red lorry, yellow lorry, red lorry, yellow lorry."),
        ("CHT_05", "The sixth sick sheik's sixth sheep is sick."),
        ("CHT_06", "Unique New York, unique New York, you know you need unique New York."),
        ("CHT_07", "Fred fed Ted bread and Ted fed Fred bread."),
        ("CHT_08", "I scream, you scream, we all scream for ice cream."),
        ("CHT_09", "Pad kid poured curd pulled cod."),
        ("CHT_10", "Whether the weather is warm, whether the weather is hot, we have to put up with the weather whether we like it or not."),
    ],
    # Long form — stability test, checks for degradation on extended passages
    'long_form': [
        ("CHL_01",
         "Artificial intelligence has transformed how we interact with technology in our daily lives. "
         "From voice assistants that answer our questions to recommendation systems that suggest what to watch next, "
         "machine learning algorithms are now deeply embedded in the tools we use most. "
         "The development of large language models has accelerated this transformation, enabling systems that can "
         "engage in sophisticated dialogue, write code, analyze documents, and perform a wide range of cognitive tasks. "
         "However, this rapid advancement raises important questions about the future of work, privacy, and the nature "
         "of intelligence itself. Researchers and policymakers are grappling with how to harness the benefits of these "
         "technologies while mitigating potential harms. The challenge is not merely technical but fundamentally social "
         "and ethical. We must decide what values we want our systems to embody, who gets to make those decisions, "
         "and how we ensure that the benefits are distributed fairly across society."),
        ("CHL_02",
         "The old lighthouse stood at the edge of the rocky cliff, its white paint faded and cracked by decades of "
         "salt air and winter storms. Every evening at dusk, the keeper would climb the worn spiral staircase to "
         "ignite the great lamp, sending its beam sweeping across the dark water in a steady, rhythmic pulse. "
         "Ships for miles around relied on that signal to navigate safely past the dangerous shoals that lay hidden "
         "beneath the surface. The keeper had inherited the post from his father, who had kept the light before him, "
         "and his grandfather before that. Three generations of faithful service, each one passing the responsibility "
         "to the next with quiet pride. The work was solitary and demanding, requiring constant vigilance through "
         "the long nights and careful maintenance during the days. But standing at the top of the tower on a clear "
         "morning, watching the sunrise paint the waves in shades of gold and amber, the keeper knew there was no "
         "other life he would have chosen."),
        ("CHL_03",
         "Climate change represents one of the most complex challenges humanity has ever faced. "
         "The science is unambiguous: global average temperatures have risen by approximately one point one degrees "
         "Celsius since pre-industrial times, driven primarily by human emissions of greenhouse gases. "
         "The consequences are already visible across the globe, from melting glaciers and rising sea levels "
         "to more frequent and intense extreme weather events. "
         "Yet despite decades of international negotiations and countless pledges to reduce emissions, "
         "global carbon dioxide output continues to rise. "
         "The gap between the commitments made by governments and the actions required to limit warming "
         "to one point five degrees Celsius remains stubbornly wide. "
         "Bridging that gap will require unprecedented cooperation across nations, industries, and communities, "
         "as well as transformative changes in how we produce energy, grow food, build cities, and move people "
         "and goods around the world."),
    ],
    # Questions — tests interrogative intonation
    'questions': [
        ("CHQ_01", "Can you tell me what the weather is like today?"),
        ("CHQ_02", "Are you sure this is the right address?"),
        ("CHQ_03", "What time does the next train to London depart?"),
        ("CHQ_04", "Have you considered all the possible consequences of this decision?"),
        ("CHQ_05", "Why did the stock market crash in nineteen twenty nine?"),
        ("CHQ_06", "Is there any chance the meeting could be rescheduled to Thursday?"),
        ("CHQ_07", "Do you know how long the surgical procedure typically takes?"),
        ("CHQ_08", "Which route would you recommend for avoiding traffic on the highway?"),
        ("CHQ_09", "Where exactly did you find that information about the experiment?"),
        ("CHQ_10", "How many people attended the annual conference last year?"),
    ],
    # Emotional — tests expressive range; urgent, sad, joyful, calm
    'emotional': [
        ("CHE_01", "I can't believe you did this to me. I am absolutely devastated."),
        ("CHE_02", "We won! We actually won! This is the greatest day of my entire life!"),
        ("CHE_03", "Please, I am begging you, this is a matter of life and death."),
        ("CHE_04", "Everything is calm and peaceful here. There is nothing at all to worry about."),
        ("CHE_05", "I am so incredibly proud of everything you have accomplished this year."),
        ("CHE_06", "The situation is completely under control. You have absolutely nothing to fear."),
        ("CHE_07", "I am furious. This is completely unacceptable behavior from any standard."),
        ("CHE_08", "What a wonderful surprise! I had absolutely no idea you were planning any of this."),
        ("CHE_09", "I'm sorry. I know that nothing I say now can undo the damage that has been done."),
        ("CHE_10", "Just breathe. Take it one step at a time. You have done this before and you can do it again."),
    ],
    # Code-switching — mixed language fragments, loanwords, foreign terms
    'code_switching': [
        ("CHX_01", "The meeting is at three PM, tres bien."),
        ("CHX_02", "She said merci beaucoup and walked gracefully out the door."),
        ("CHX_03", "The concept of schadenfreude is notoriously difficult to translate into English."),
        ("CHX_04", "He said auf Wiedersehen and boarded the train to Frankfurt."),
        ("CHX_05", "The menu listed a prix fixe dinner for sixty euros per person."),
        ("CHX_06", "Use the resume template for all new job applicants going forward."),
        ("CHX_07", "The esprit de corps of the team was remarkable throughout the entire campaign."),
        ("CHX_08", "Her joie de vivre was absolutely infectious to everyone around her."),
        ("CHX_09", "The company reported strong third quarter results, a true coup de grace for investors."),
        ("CHX_10", "The zeitgeist of the era was captured perfectly in that singular photograph."),
    ],
}

# Legacy sample texts (kept for backward compatibility)
SAMPLE_TEXTS = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
    "The small pup gnawed a hole in the sock.",
    "The fish twisted and turned on the bent hook.",
    "Press the pants and sew a button on the vest.",
    "The swan dive was far short of perfect.",
    "The beauty of the view stunned the young boy.",
    "Two blue fish swam in the tank.",
    "Her purse was full of useless trash.",
    "The colt reared and threw the tall rider.",
    "It snowed, rained, and hailed the same morning.",
    "Read verse out loud for pleasure.",
]


# ── Dataset builders ───────────────────────────────────────────────────────────

def create_sample_manifest(n_samples: int, seed: int) -> List[Dict]:
    """Create sample manifest for testing without real dataset."""
    random.seed(seed)

    manifest = []
    for i in range(n_samples):
        text = random.choice(SAMPLE_TEXTS)
        speaker_id = f"spk_{i % 50:03d}"

        manifest.append({
            'id': f'SAMPLE_{i:05d}',
            'text': text,
            'dataset': 'seed_tts_eval_sample',
            'category': 'sample',
            'language': 'en',
            'difficulty': 'standard',
            'reference_audio_path': None,
            'speaker_id': speaker_id
        })

    return manifest


def create_harvard_manifest(n_samples: int = None, seed: int = 42) -> List[Dict]:
    """Create manifest from Harvard Sentences (IEEE 1969).

    Args:
        n_samples: If set, randomly sample this many sentences; otherwise use all 100.
        seed: Random seed for sampling reproducibility.
    """
    entries = []
    for uid, text in HARVARD_SENTENCES:
        entries.append({
            'id': f'HAR_{uid}',
            'text': text,
            'dataset': 'harvard_sentences',
            'category': 'phonetically_balanced',
            'language': 'en',
            'difficulty': 'standard',
            'reference_audio_path': None,
            'speaker_id': 'spk_harvard',
        })

    if n_samples is not None and n_samples < len(entries):
        random.seed(seed)
        entries = random.sample(entries, n_samples)

    return entries


def create_challenge_manifest(
    categories: List[str] = None,
    n_samples_per_category: int = None,
    seed: int = 42,
) -> List[Dict]:
    """Create manifest from Custom Challenge Set.

    Args:
        categories: Which categories to include. None = all.
        n_samples_per_category: If set, sample this many per category.
        seed: Random seed for reproducibility.
    """
    if categories is None:
        categories = list(CHALLENGE_TEXTS.keys())

    # Difficulty mapping
    difficulty_map = {
        'numbers': 'hard',
        'proper_nouns': 'hard',
        'tongue_twisters': 'hard',
        'long_form': 'very_hard',
        'questions': 'standard',
        'emotional': 'standard',
        'code_switching': 'hard',
    }

    entries = []
    random.seed(seed)

    for cat in categories:
        if cat not in CHALLENGE_TEXTS:
            print(f"  ⚠ Unknown category '{cat}', skipping.")
            continue

        items = list(CHALLENGE_TEXTS[cat])
        if n_samples_per_category is not None and n_samples_per_category < len(items):
            items = random.sample(items, n_samples_per_category)

        for uid, text in items:
            entries.append({
                'id': f'CHL_{uid}',
                'text': text,
                'dataset': f'challenge_{cat}',
                'category': cat,
                'language': 'en',
                'difficulty': difficulty_map.get(cat, 'standard'),
                'reference_audio_path': None,
                'speaker_id': f'spk_{cat}',
            })

    return entries


def parse_real_dataset(dataset_path: Path) -> List[Dict]:
    """Parse real Seed-TTS-Eval dataset if available."""
    meta_files = list(dataset_path.glob("**/en/meta.lst"))

    if not meta_files:
        return []

    meta_file = meta_files[0]
    entries = []

    print(f"▶ Parsing {meta_file}")

    with open(meta_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('|')

            if len(parts) >= 4:
                filename = parts[0].strip()
                prompt_audio = parts[2].strip() if len(parts) > 2 else None
                text_to_synth = parts[3].strip() if len(parts) > 3 else parts[1].strip()

                ref_audio_path = None
                if prompt_audio:
                    ref_path = dataset_path / "en" / prompt_audio
                    if ref_path.exists():
                        ref_audio_path = str(ref_path)

                entries.append({
                    'id': f'SEED_{idx:05d}',
                    'text': text_to_synth,
                    'dataset': 'seed_tts_eval',
                    'category': 'tts',
                    'language': 'en',
                    'difficulty': 'standard',
                    'reference_audio_path': ref_audio_path,
                    'speaker_id': f'spk_{idx % 100:03d}'
                })

    return entries


def create_ljspeech_manifest(
    dataset_path: Path,
    n_samples: int = None,
    seed: int = 42,
) -> List[Dict]:
    """Create manifest from LJ Speech dataset (single speaker, studio quality).

    Preferred — stream only what you need via HuggingFace (no full download):
      python datasets/download.py --dataset ljspeech --n-samples 100

    Full download (2.6 GB, gives access to all 13,100 clips):
      wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
      tar -xjf LJSpeech-1.1.tar.bz2 -C datasets/

    Args:
        dataset_path: Path to LJSpeech-1.1/ directory.
        n_samples: Number of utterances to use. If the local dataset is absent,
                   streams exactly this many from HuggingFace (default: 100).
        seed: Random seed for sampling when using local data.
    """
    metadata_path = dataset_path / "metadata.csv"

    # ── Local dataset present ─────────────────────────────────────────────────
    if metadata_path.exists():
        entries = []
        with open(metadata_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 3:
                    continue
                uid, _, normalized_text = parts[0], parts[1], parts[2]
                wav_path = dataset_path / "wavs" / f"{uid}.wav"
                if not wav_path.exists():
                    continue
                entries.append({
                    'id': f'LJS_{uid}',
                    'text': normalized_text,
                    'dataset': 'ljspeech',
                    'category': 'studio_quality',
                    'language': 'en',
                    'difficulty': 'standard',
                    'reference_audio_path': str(wav_path),
                    'speaker_id': 'lj_single',
                })

        if n_samples is not None and n_samples < len(entries):
            random.seed(seed)
            entries = random.sample(entries, n_samples)

        print(f"  ✓ LJ Speech (local): {len(entries)} entries")
        return entries

    # ── Stream from HuggingFace — downloads only N samples ───────────────────
    n = n_samples if n_samples is not None else 100
    print(f"  ▶ Streaming {n} LJ Speech samples from HuggingFace "
          f"(no full download needed)...")
    return _stream_ljspeech_hf(n, dataset_path.parent / "ljspeech_hf_audio")


def _stream_ljspeech_hf(n_samples: int, audio_save_dir: Path) -> List[Dict]:
    """Stream n_samples from keithito/lj_speech on HuggingFace."""
    try:
        import soundfile as sf
        import numpy as np
        from datasets import load_dataset
    except ImportError:
        print("  ✗ Missing packages. Run: pip install datasets soundfile")
        return []

    audio_save_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(
            "keithito/lj_speech",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ✗ Could not load LJ Speech from HuggingFace: {e}")
        print("    Try: pip install datasets")
        return []

    entries = []
    for item in ds:
        if len(entries) >= n_samples:
            break

        uid = item['id']
        text = item.get('normalized_text') or item.get('text', '')
        audio_info = item.get('audio', {})
        array = audio_info.get('array')
        sr = audio_info.get('sampling_rate', 22050)

        if array is None or not text:
            continue

        wav_path = audio_save_dir / f"{uid}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), np.array(array), sr)

        entries.append({
            'id': f'LJS_{uid}',
            'text': text,
            'dataset': 'ljspeech',
            'category': 'studio_quality',
            'language': 'en',
            'difficulty': 'standard',
            'reference_audio_path': str(wav_path),
            'speaker_id': 'lj_single',
        })

        if len(entries) % 10 == 0:
            print(f"    {len(entries)}/{n_samples} downloaded...", end='\r')

    print(f"  ✓ LJ Speech (HuggingFace stream): {len(entries)} entries "
          f"saved to {audio_save_dir}")
    return entries


def _stream_vctk_hf(
    n_speakers: int,
    n_per_speaker: int,
    seed: int,
    audio_save_dir: Path,
) -> List[Dict]:
    """Stream VCTK samples from HuggingFace (speech-recognition/vctk)."""
    try:
        import soundfile as sf
        import numpy as np
        from datasets import load_dataset
    except ImportError:
        print("  ✗ Missing packages. Run: pip install datasets soundfile")
        return []

    audio_save_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    try:
        ds = load_dataset(
            "speech-recognition/vctk",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ✗ Could not load VCTK from HuggingFace: {e}")
        return []

    # Collect up to n_per_speaker per speaker, stopping at n_speakers
    by_speaker: Dict[str, list] = {}
    for item in ds:
        spk = str(item.get('speaker_id', item.get('speaker', 'unknown')))
        if spk not in by_speaker:
            if len(by_speaker) >= n_speakers:
                # Check if all speakers are full
                if all(len(v) >= n_per_speaker for v in by_speaker.values()):
                    break
                continue  # skip new speaker if we have enough
            by_speaker[spk] = []

        if len(by_speaker[spk]) >= n_per_speaker:
            continue

        text = item.get('text', item.get('sentence', ''))
        audio_info = item.get('audio', {})
        array = audio_info.get('array')
        sr = audio_info.get('sampling_rate', 16000)
        uid = item.get('file', item.get('id', f"{spk}_{len(by_speaker[spk]):03d}"))
        uid = Path(uid).stem  # strip extension if present

        if array is None or not text:
            continue

        wav_path = audio_save_dir / f"{uid}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), np.array(array), sr)

        by_speaker[spk].append({
            'id': f'VCTK_{spk}_{uid}',
            'text': text,
            'dataset': 'vctk',
            'category': 'multi_speaker',
            'language': 'en',
            'difficulty': 'standard',
            'reference_audio_path': str(wav_path),
            'speaker_id': spk,
        })

        total = sum(len(v) for v in by_speaker.values())
        if total % 10 == 0:
            print(f"    {total}/{n_speakers * n_per_speaker} downloaded...", end='\r')

    entries = [e for spk_entries in by_speaker.values() for e in spk_entries]
    n_spk = len(by_speaker)
    print(f"  ✓ VCTK (HuggingFace stream): {len(entries)} entries "
          f"({n_spk} speakers) saved to {audio_save_dir}")
    return entries


def create_vctk_manifest(
    dataset_path: Path,
    n_speakers: int = 10,
    n_per_speaker: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """Create manifest from VCTK dataset (110 speakers, various UK accents).

    Preferred — stream only what you need via HuggingFace:
      python datasets/download.py --dataset vctk --vctk-speakers 10 --vctk-per-speaker 10

    Full download (11 GB):
      wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip
      unzip DS_10283_3443.zip -d datasets/vctk/

    Args:
        dataset_path: Path to VCTK-Corpus/ or VCTK-Corpus-0.92/ directory.
        n_speakers: Number of speakers to include.
        n_per_speaker: Number of utterances per speaker.
        seed: Random seed.
    """
    # VCTK has two possible wav directory names
    wav_dir = None
    for candidate in ['wav48_silence_trimmed', 'wav48', 'wav16']:
        if (dataset_path / candidate).exists():
            wav_dir = dataset_path / candidate
            break

    txt_dir = dataset_path / "txt"

    if wav_dir is None or not txt_dir.exists():
        print(f"  ✗ VCTK not found at {dataset_path} — streaming from HuggingFace instead...")
        return _stream_vctk_hf(
            n_speakers=n_speakers,
            n_per_speaker=n_per_speaker,
            seed=seed,
            audio_save_dir=Path('datasets/vctk_hf_audio'),
        )

    random.seed(seed)
    speakers = sorted([d.name for d in txt_dir.iterdir() if d.is_dir()])
    if n_speakers < len(speakers):
        speakers = random.sample(speakers, n_speakers)

    entries = []
    for spk in sorted(speakers):
        spk_wav_dir = wav_dir / spk
        spk_txt_dir = txt_dir / spk
        if not spk_wav_dir.exists() or not spk_txt_dir.exists():
            continue

        # Collect matched text+audio pairs
        pairs = []
        for txt_file in sorted(spk_txt_dir.glob("*.txt")):
            uid = txt_file.stem
            # wav may be .wav or .flac
            wav_file = None
            for ext in ['.wav', '.flac', '_mic1.flac', '_mic2.flac']:
                candidate = spk_wav_dir / f"{uid}{ext}"
                if candidate.exists():
                    wav_file = candidate
                    break
            if wav_file is None:
                continue
            text = txt_file.read_text(encoding='utf-8').strip()
            if text:
                pairs.append((uid, text, wav_file))

        if n_per_speaker < len(pairs):
            pairs = random.sample(pairs, n_per_speaker)

        for uid, text, wav_file in pairs:
            entries.append({
                'id': f'VCTK_{spk}_{uid}',
                'text': text,
                'dataset': 'vctk',
                'category': 'multi_speaker',
                'language': 'en',
                'difficulty': 'standard',
                'reference_audio_path': str(wav_file),
                'speaker_id': spk,
            })

    print(f"  ✓ VCTK: {len(entries)} entries ({len(speakers)} speakers × ~{n_per_speaker} utterances)")
    return entries


def create_cmu_arctic_manifest(
    dataset_root: Path,
    speakers: List[str] = None,
    n_per_speaker: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """Create manifest from CMU Arctic dataset (multi-speaker, phonetically balanced).

    Download speakers first:
      wget http://www.festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2
      wget http://www.festvox.org/cmu_arctic/packed/cmu_us_bdl_arctic.tar.bz2
      tar -xjf cmu_us_slt_arctic.tar.bz2 -C datasets/cmu_arctic/
      tar -xjf cmu_us_bdl_arctic.tar.bz2 -C datasets/cmu_arctic/

    Args:
        dataset_root: Directory containing cmu_us_*/  speaker folders.
        speakers: Speaker IDs to include (e.g. ['slt', 'bdl']). None = all found.
        n_per_speaker: Utterances per speaker.
        seed: Random seed.
    """
    if not dataset_root.exists():
        print(f"  ✗ CMU Arctic not found at {dataset_root}")
        print("    Download from: http://www.festvox.org/cmu_arctic/packed/")
        return []

    # Find speaker directories
    spk_dirs = sorted(dataset_root.glob("cmu_us_*_arctic"))
    if speakers:
        spk_dirs = [d for d in spk_dirs if any(s in d.name for s in speakers)]

    if not spk_dirs:
        print(f"  ✗ No CMU Arctic speaker directories found in {dataset_root}")
        return []

    random.seed(seed)
    entries = []

    for spk_dir in spk_dirs:
        spk_id = spk_dir.name.replace('cmu_us_', '').replace('_arctic', '')

        # Parse text file
        txt_file = spk_dir / "etc" / "txt.done.data"
        if not txt_file.exists():
            continue

        pairs = []
        with open(txt_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('('):
                    continue
                # Format: ( arctic_a0001 "text here" )
                parts = line.strip('()').split('"')
                if len(parts) < 2:
                    continue
                uid = parts[0].strip()
                text = parts[1].strip()
                wav_path = spk_dir / "wav" / f"{uid}.wav"
                if wav_path.exists() and text:
                    pairs.append((uid, text, wav_path))

        if n_per_speaker < len(pairs):
            pairs = random.sample(pairs, n_per_speaker)

        for uid, text, wav_path in pairs:
            entries.append({
                'id': f'ARCTIC_{spk_id}_{uid}',
                'text': text,
                'dataset': 'cmu_arctic',
                'category': 'phonetically_balanced',
                'language': 'en',
                'difficulty': 'standard',
                'reference_audio_path': str(wav_path),
                'speaker_id': spk_id,
            })

    n_spk = len(set(e['speaker_id'] for e in entries))
    print(f"  ✓ CMU Arctic: {len(entries)} entries ({n_spk} speakers)")
    return entries


def create_multi_dataset_manifest(
    datasets: List[str],
    n_samples: int = None,
    seed: int = 42,
    real_dataset_path: Path = None,
    challenge_categories: List[str] = None,
) -> List[Dict]:
    """Combine multiple dataset manifests into one unified manifest.

    Args:
        datasets: List of datasets to include: 'seed', 'harvard', 'challenge'
        n_samples: If set, limit total entries via proportional sampling.
        seed: Random seed.
        real_dataset_path: Path to real Seed-TTS-Eval data (optional).
        challenge_categories: Which challenge categories to include (None = all).
    """
    all_entries = []

    # --- Seed-TTS-Eval ---
    if 'seed' in datasets:
        if real_dataset_path and real_dataset_path.exists():
            entries = parse_real_dataset(real_dataset_path)
            if not entries:
                print("  ✗ Could not parse real dataset, falling back to sample data.")
                entries = create_sample_manifest(50, seed)
        else:
            # Proportional share: ~50 sample utterances when combined
            sample_n = 50 if n_samples else 200
            entries = create_sample_manifest(sample_n, seed)
        all_entries.extend(entries)
        print(f"  ✓ Seed-TTS-Eval: {len(entries)} entries")

    # --- Harvard Sentences ---
    if 'harvard' in datasets:
        entries = create_harvard_manifest(seed=seed)  # All 100
        all_entries.extend(entries)
        print(f"  ✓ Harvard Sentences: {len(entries)} entries")

    # --- Challenge Set ---
    if 'challenge' in datasets:
        entries = create_challenge_manifest(
            categories=challenge_categories,
            seed=seed,
        )
        all_entries.extend(entries)
        print(f"  ✓ Challenge Set: {len(entries)} entries "
              f"({len(set(e['category'] for e in entries))} categories)")

    # Global sampling if requested
    if n_samples is not None and n_samples < len(all_entries):
        random.seed(seed)
        all_entries = random.sample(all_entries, n_samples)
        print(f"  ✓ Sampled {n_samples} entries total (seed={seed})")

    return all_entries


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare TTS evaluation datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with sample sentences (default):
  python datasets/download.py
  # → saves to datasets/seed_manifest.json

  # All Harvard Sentences:
  python datasets/download.py --dataset harvard
  # → saves to datasets/harvard_manifest.json

  # Challenge set, numbers + tongue twisters only:
  python datasets/download.py --dataset challenge --categories numbers tongue_twisters
  # → saves to datasets/challenge_manifest.json

  # Everything combined (recommended for full eval):
  python datasets/download.py --dataset all
  # → saves to datasets/all_manifest.json

  # Override save location (e.g. set as the active manifest):
  python datasets/download.py --dataset harvard --output datasets/manifest.json

  # Real Seed-TTS-Eval + Harvard + Challenge:
  python datasets/download.py --dataset all --use-real-data --n-samples 300
""")

    parser.add_argument(
        '--dataset', type=str, default='seed',
        choices=['seed', 'harvard', 'challenge', 'all',
                 'ljspeech', 'vctk', 'cmu_arctic'],
        help='Dataset(s) to prepare (default: seed)')
    parser.add_argument(
        '--categories', nargs='+',
        choices=list(CHALLENGE_TEXTS.keys()),
        default=None,
        help='Challenge categories to include (default: all). '
             'Only used when --dataset is challenge or all.')
    parser.add_argument(
        '--n-samples', type=int, default=None,
        help='Maximum total samples (default: include all). '
             'For --dataset seed, defaults to 200 when not set.')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)')
    parser.add_argument(
        '--use-real-data', action='store_true',
        help='Use real Seed-TTS-Eval data (must be downloaded first)')
    parser.add_argument(
        '--ljspeech-path', type=str, default='datasets/LJSpeech-1.1',
        help='Path to LJSpeech-1.1/ directory (default: datasets/LJSpeech-1.1)')
    parser.add_argument(
        '--vctk-path', type=str, default='datasets/vctk/VCTK-Corpus',
        help='Path to VCTK-Corpus/ directory (default: datasets/vctk/VCTK-Corpus)')
    parser.add_argument(
        '--vctk-speakers', type=int, default=10,
        help='Number of VCTK speakers to include (default: 10)')
    parser.add_argument(
        '--vctk-per-speaker', type=int, default=10,
        help='Utterances per VCTK speaker (default: 10)')
    parser.add_argument(
        '--cmu-arctic-path', type=str, default='datasets/cmu_arctic',
        help='Path to CMU Arctic root directory (default: datasets/cmu_arctic)')
    parser.add_argument(
        '--cmu-speakers', nargs='+', default=None,
        help='CMU Arctic speaker IDs (e.g. slt bdl). Default: all found.')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for manifest JSON. '
             'Defaults to datasets/{dataset}_manifest.json to avoid overwriting. '
             'Use "datasets/manifest.json" to use as active manifest.')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    real_dataset_path = script_dir / "seed_tts_eval_data"

    # ── Build manifest ────────────────────────────────────────────────────────

    print(f"▶ Preparing dataset: {args.dataset}")

    if args.dataset == 'all':
        datasets_to_use = ['seed', 'harvard', 'challenge']
    else:
        datasets_to_use = [args.dataset]

    use_real = args.use_real_data
    if use_real and not real_dataset_path.exists():
        print("  ✗ Real dataset not found at datasets/seed_tts_eval_data/")
        print("    Falling back to sample data.")
        use_real = False

    if 'seed' in datasets_to_use and len(datasets_to_use) == 1:
        # Pure seed mode: use legacy behaviour with n-samples default
        n_samples = args.n_samples if args.n_samples else 200
        if use_real:
            entries = parse_real_dataset(real_dataset_path)
            if not entries:
                print("  ✗ Could not parse real dataset, using sample data.")
                entries = create_sample_manifest(n_samples, args.seed)
            elif n_samples < len(entries):
                random.seed(args.seed)
                entries = random.sample(entries, n_samples)
        else:
            entries = create_sample_manifest(n_samples, args.seed)
        print(f"  ✓ {len(entries)} entries from seed dataset")

    elif len(datasets_to_use) == 1 and datasets_to_use[0] == 'harvard':
        n = args.n_samples  # None = all 100
        entries = create_harvard_manifest(n_samples=n, seed=args.seed)
        print(f"  ✓ {len(entries)} Harvard Sentences")

    elif len(datasets_to_use) == 1 and datasets_to_use[0] == 'challenge':
        entries = create_challenge_manifest(
            categories=args.categories,
            seed=args.seed,
        )
        print(f"  ✓ {len(entries)} challenge entries")

    elif args.dataset == 'ljspeech':
        entries = create_ljspeech_manifest(
            dataset_path=Path(args.ljspeech_path),
            n_samples=args.n_samples,
            seed=args.seed,
        )

    elif args.dataset == 'vctk':
        entries = create_vctk_manifest(
            dataset_path=Path(args.vctk_path),
            n_speakers=args.vctk_speakers,
            n_per_speaker=args.vctk_per_speaker,
            seed=args.seed,
        )

    elif args.dataset == 'cmu_arctic':
        entries = create_cmu_arctic_manifest(
            dataset_root=Path(args.cmu_arctic_path),
            speakers=args.cmu_speakers,
            seed=args.seed,
        )

    else:
        # Multi-dataset
        real_path = real_dataset_path if use_real else None
        entries = create_multi_dataset_manifest(
            datasets=datasets_to_use,
            n_samples=args.n_samples,
            seed=args.seed,
            real_dataset_path=real_path,
            challenge_categories=args.categories,
        )

    # ── Save manifest ─────────────────────────────────────────────────────────
    if args.output:
        manifest_path = Path(args.output)
    else:
        # Auto-name by dataset so runs don't overwrite each other
        # e.g. harvard_manifest.json, challenge_manifest.json, all_manifest.json
        manifest_path = script_dir / f"{args.dataset}_manifest.json"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(entries, f, indent=2)

    # Summary
    print(f"\n✓ Saved manifest to {manifest_path}")
    print(f"✓ Total entries: {len(entries)}")

    # Dataset breakdown
    by_ds: Dict[str, int] = {}
    by_cat: Dict[str, int] = {}
    for e in entries:
        ds = e.get('dataset', 'unknown')
        cat = e.get('category', 'unknown')
        by_ds[ds] = by_ds.get(ds, 0) + 1
        by_cat[cat] = by_cat.get(cat, 0) + 1

    if len(by_ds) > 1:
        print("\nDataset breakdown:")
        for ds, cnt in sorted(by_ds.items()):
            print(f"  {ds:35s}  {cnt:4d}")

    if len(by_cat) > 1:
        print("\nCategory breakdown:")
        for cat, cnt in sorted(by_cat.items()):
            print(f"  {cat:30s}  {cnt:4d}")

    if entries:
        print("\nExample entry:")
        print(json.dumps(entries[0], indent=2))

    # Hint about the challenge categories
    if 'challenge' not in datasets_to_use:
        print("\n" + "="*60)
        print("TIP: Add --dataset challenge to include the Custom Challenge Set:")
        print("  numbers, proper_nouns, tongue_twisters, long_form,")
        print("  questions, emotional, code_switching")
        print("="*60)


if __name__ == '__main__':
    main()
