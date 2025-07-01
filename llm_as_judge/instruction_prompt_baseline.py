class InstructionPromptBaseline:
    
    high_level_instruction = '''These are statements taken from people’s conversations during Diplomacy games played online. Diplomacy is a game about pre-World War 1 Europe. It usually has seven players: England, France, Germany, Italy, Austria-Hungary, Russia, and Turkey.

In these statements, players try to form alliances to plan military campaigns and defeat each other, but things might change quickly.

Each statement is a piece of a dialogue from a SENDER player to a RECEIVER player.

Please classify the statements according to whether the sender is talking about game moves, other players, reasoning out a move, or trying to build a rapport with the receiver.

Select "YES" if you're really confident about your answer. A single statement can have a "YES" for more than one question.

Underlined words suggest what to look out for, but there will be other signals too.'''

    question_answer = '''For each of the following questions, answer "YES" if you are confident about your answer. A single statement can have a "YES" for more than one question. Underlined words suggest what to look out for, but there will be other signals too.

**1. Is this statement about the sender's or receiver's GAME MOVE?**

The sender states an actual or suggested game move by the sender or the receiver. It might also be in the form of an acceptance, a question, or a clarification.

**2. Does this statement PROVIDE REASONS for the sender's or receiver's move?**

The sender offers justification or explanations for a move by themselves or by the receiver, guesses what moves might happen next, or discusses a move that already happened.

**3. Does this statement involve BUILDING a RAPPORT?**

In this statement, the sender wants to build a rapport with the receiver through "you and me" dialogue and personal information sharing.

**4. Is the sender greeting or paying a COMPLIMENT to the receiver?**

The sender is greeting or paying a compliment to the receiver.

**5. Is the sender offering REASSURANCE to the receiver?**

The sender is reassuring the receiver.

**6. Is the sender APOLOGISING to the receiver?**

The sender is apologising to the receiver.

**7. Is the sender SHARING PERSONAL THOUGHTS or feelings with the receiver?**

The sender is sharing their personal thoughts or feelings with the receiver.

**8. Does this statement SHARE INFORMATION about other players?**

This statement shares information related to other game players, NOT the sender or the receiver.

'''

    def final_prompt(self, statement):
        prompt = ''
        prompt += self.high_level_instruction
        prompt += f'''Please answer "YES" if you're really confident about your answer. A single statement can have a "YES" for more than one question. Underlined words suggest what to look out for, but there will be other signals too. Carefully read the statement and answer the questions one by one. Please only give me the answer with the question number and only answer YES/NO for each question (total 8 questions) without any other text or explanation.

The final format should be like:
```
1. YES
2. NO
...
...
```

Here is the statement:

{statement}

    '''
        return prompt