class InstructionPromptFewShot:
    
    fewshot_examples = [
    {
        "text": "I really don't think an attack from Gascony would be an issue, but I understand that your trust in France is much lower than mine.",
        "labels": ["YES", "YES", "YES", "NO", "NO", "YES", "YES", "NO"]
    },
    {
        "text": "Either way, telling me that is not good news for me",
        "labels": ["NO", "NO", "NO", "NO", "NO", "YES", "NO", "YES"]
    },
    {
        "text": "I don't have much info right now unfortunately.",
        "labels": ["NO", "NO", "NO", "YES", "NO", "YES", "NO", "NO"]
    }
]
    
    high_level_instruction = '''These are statements taken from people’s conversations during Diplomacy games played online. Diplomacy is a game about pre-World War 1 Europe. It usually has seven players: England, France, Germany, Italy, Austria-Hungary, Russia, and Turkey.

In these statements, players try to form alliances to plan military campaigns and defeat each other, but things might change quickly.

Each statement is a piece of a dialogue from a SENDER player to a RECEIVER player.

Please classify the statements according to whether the sender is talking about game moves, other players, reasoning out a move, or trying to build a rapport with the receiver.

Select "YES" if you're really confident about your answer. A single statement can have a "YES" for more than one question.

Underlined words suggest what to look out for, but there will be other signals too.'''

    overview = '''### Overview
In this job, you will be presented with a statement made during an online Diplomacy game. The statement is made by one player to another. It usually discusses the next move and why to make it. Sometimes it is simply a friendly exchange between two players.

Review the text of the statement and help us by answering a few yes/no questions about it. Each HIT takes about 2 minutes.
    '''

    steps = '''### Steps
- Read the statement.
- Determine which category best describe the statement. 
    '''

    rules_tips = '''### Rules & Tips
- GAME MOVE Description: 
    - YES: This statement is about an actual or suggested game move by **the sender or the receiver**. It might also be in the form of an acceptance, a question, or a clarification.
    ("**I'm attempting to** make that deal with russia now"; "**I am committed to** supporting Munich holding"; "**Make sure you don't** move Munich so that it can take my support."; "Why would France help us?" ;"Why would you want me to strike war with England?"; “**I think the most important thing for you** right now is getting England fully committed against France”; "I suggest that you order: Kiel support Berlin holding..."; "**I strongly recommend** it.")
    - NO: This statement is not describing a game move.  


- PROVIDE REASONS Description:  
    - YES: This statement uses the point-of-views of the sender or the receiver **to justify a move, guess what moves might happen next, or discuss a move that already happened.**
    ("If you took Marseilles, **I would be stronger** against England"; "I think France **will try to influence you**"; “You **could have advised me** that supporting Mun-Bur was more important than Kie-Ruh”; "France is a really good player, and he is no doubt working hard **to get England to turn on you.**”)
    - NO: This statement does not provide reasons for the sender's or receiver's move.


- BUILDING A RAPPORT Description:  
    - YES: In this statement, the sender wants to build a rapport with the receiver through "you and me" dialogue: either through **compliments, sharing honest concerns, reassurances, or apologies.**
    (“Let's keep it **between you and me**!"; "I won't hold it against you"; "You're **my favorite**."; "Sure. But, **you'll see from my moves** this turn that Austria is lying to you."; "**I mean it** sincerely."; "**I'd much rather work with you**."; "**We'll crack this** eventually.”; "**I'm going to keep helping you** as much as I can.")
    - NO: This statement does not appear to build a relationship.


- WAYS TO BUILD RAPPORT Description:
If the statement is building a rapport, please tell us how it is doing so.
    - COMPLIMENT: In this statement, the sender is greeting or paying a compliment to the reciever.
    ("Good day to you Germany!")
    - REASSURANCE: In this statement, the sender is reassuring the receiver.
    ("I promise I'll never let you down")
    - APOLOGIES: In this statement, the sender is apologising to the receiver.
    ("I should've let you know")
    - PERSONAL THOUGHTS: In this statement, the sender is expressing his personal thoughts to the receiver.
    ("Let's Keep it between you and me!")
    
- SHARING INFORMATION Description:  
    - YES: This statement shares information related to **other game players, NOT** the sender or the receiver.
    (“**France is a really good player**, and he is no doubt working hard to get England to turn on you.”; "**England's pieces** are not positioned well if he's trying to attack France or contain Italy."; "**France held out a long time**"; "And anything that's bad for Russia right now **is good for Austria.**")
    - NO: This statement does not share information related to other game players.
    '''
    
    question_answer = '''For each of the following questions, answer "YES" if you are confident about your answer. A single statement can have a "YES" for more than one question. Underlined words suggest what to look out for, but there will be other signals too.

**1. Is this statement about the sender's or receiver's GAME MOVE?**

The sender states an actual or suggested game move by the sender or the receiver. It might also be in the form of an acceptance, a question, or a clarification.

Examples:
- "I'm attempting to make that deal with Russia now."
- "I am committed to supporting Munich holding."
- "Make sure you don't move Munich so that it can take my support."
- "I think the most important thing for you right now is getting England fully committed against France."
- "I suggest that you order: Kiel support Berlin holding..."
- "I strongly recommend it."

**2. Does this statement PROVIDE REASONS for the sender's or receiver's move?**

The sender offers justification or explanations for a move by themselves or by the receiver, guesses what moves might happen next, or discusses a move that already happened.

Examples:
- "If you took Marseilles, I would be stronger against England."
- "I think France will try to influence you."
- "You could have advised me that supporting Mun-Bur was more important than Kie-Ruh."
- "France is a really good player, and he is no doubt working hard to get England to turn on you."

**3. Does this statement involve BUILDING a RAPPORT?**

In this statement, the sender wants to build a rapport with the receiver through "you and me" dialogue and personal information sharing.

Examples:
- "Great to hear. Thank you."
- "Thanks, I'll work on these."
- "Let's keep it between you and me!"
- "I won't hold it against you."
- "You're my favorite."
- "We'll crack this eventually."
- "I'm going to keep helping you as much as I can."
- "But in the interest of continued full disclosure, here's what I think."
- "So I'm in sort of a conflicted spot."

**4. Is the sender greeting or paying a COMPLIMENT to the receiver?**

The sender is greeting or paying a compliment to the receiver.

Examples:
- "Good day to you Germany!"
- "Thanks Italy. Hope you're enjoying the weather on the Anatolian."
- "Your logic is undeniable. Enjoy your stay in Tyr!"
- "You are my favorite."
- "Okay, can do. Thanks!"

**5. Is the sender offering REASSURANCE to the receiver?**

The sender is reassuring the receiver.

Examples:
- "I promise I'll never let you down."
- "I won't hold it against you."
- "Sure. But, you'll see from my moves this turn that Austria is lying to you."
- "I mean it sincerely."
- "I'd much rather work with you."
- "We'll crack this eventually."
- "I'm going to keep helping you as much as I can."

**6. Is the sender APOLOGISING to the receiver?**

The sender is apologising to the receiver.

Examples:
- "Sorry I won't be able to cut off Gascony this turn..."
- "Okay sorry for being nosy! I will try for Bur on the off chance it shakes out that way."
- "Ha! So sorry!! I meant that for France!"
- "I should've let you know."

**7. Is the sender SHARING PERSONAL THOUGHTS or feelings with the receiver?**

The sender is sharing their personal thoughts or feelings with the receiver.

Examples:
- "Let's keep it between you and me!"
- "I like to coordinate, but on these sort of 50/50 guesses, I kind of like to keep it secret so that if it doesn’t go well, I have nobody to blame but myself."
- "Okay, so I still have a teensy little bone to pick with you: on the off-chance that Austria wasn't lying and you *did* take Trieste unexpectedly, I sort of worry that I might be next."
- "I have some thoughts on the matter, and some information, but I'd like to feel confident that you and I will keep anything we say between us."
- "But in the interest of continued full disclosure, here's what I think."
- "So I'm in sort of a conflicted spot."

**8. Does this statement SHARE INFORMATION about other players?**

This statement shares information related to other game players, NOT the sender or the receiver.

Examples:
- "France is a really good player, and he is no doubt working hard to get England to turn on you."
- "England's pieces are not positioned well if he's trying to attack France or contain Italy."
- "France held out a long time."
- "And anything that's bad for Russia right now is good for Austria."
'''

    def final_prompt(self, statement):
        prompt = ''
        prompt += self.high_level_instruction
        prompt += self.overview
        prompt += self.steps
        prompt += self.rules_tips
        prompt += self.question_answer
        prompt += f'''Please answer "YES" if you're really confident about your answer. A single statement can have a "YES" for more than one question. Underlined words suggest what to look out for, but there will be other signals too. Carefully read the statement and answer the questions one by one. Please only give me the answer with the question number and only answer YES/NO for each question (total 8 questions) without any other text or explanation.\n\nHere are some examples:\n\n"""\n'''
        for ex in self.fewshot_examples:
            prompt += f"""Statement:
{ex['text']}
""" + "\n".join(f"{idx}. {ans}" for idx, ans in enumerate(ex["labels"], start=1)) + "\n\n"
        prompt += '"""\n\n'
        prompt += f'''Please make sure your thinking chain is in <think> ... </think> tag, think about it step by step.

The final format should be like:
```
<think> ... </think>
1. YES
2. NO
...
...
```

Here is the statement:

{statement}
    '''
        return prompt
