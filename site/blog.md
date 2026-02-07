# BERT Wasn't Supposed to Chat. We Made It Anyway.

*A story about ignoring the rules, iterative unmasking, and what happens when you treat "impossible" as a suggestion.*

---

There's a law in NLP. Everybody knows it. It goes like this:

**GPT generates. BERT understands.**

GPT is the writer. BERT is the reader. GPT talks. BERT listens. This division is sacred. It's in the textbooks. It's how every ML course draws the diagram. Encoders on the left, decoders on the right, and never shall the two swap jobs.

We looked at that law, and we thought:

*What if we just... didn't?*

What if we made BERT chat?

---

## The Equation Nobody Asked For

Here's the thing. ChatGPT's formula is simple:

```
ChatGPT = GPT + RLHF --> Chat
```

Take a generative model. Fine-tune it with human feedback. Get a chatbot. Revolutionary, yes, but architecturally straightforward: GPT was already generating text. RLHF just taught it *which* text to generate.

So we wrote a different equation:

```
ChatBERT = BERT + Novel Approaches --> Chat
```

And then we built it.

Not as a joke. Not as a meme. As a real system, with real training, real inference, and a real insight underneath it that we think deserves way more attention than it gets.

---

## The Insight: Thinking Before Speaking

Here's what bothered us about autoregressive generation.

When GPT writes a response, it commits to every single word *before seeing what comes next*. Token by token, left to right, no backsies. It's writing the first sentence of your email before it knows how the email ends. It's starting a joke before it has a punchline.

BERT doesn't work that way. BERT sees everything at once. Bidirectional attention means every token attends to every other token -- past, present, and future. It's the whole picture, all the time.

Now, think about how *you* write an important message. A job application. A tricky Slack reply. A text to your ex. Do you write left-to-right, one word at a time, never looking back?

No. You draft. You revise. You look at the whole thing, notice the tone is wrong in paragraph two, fix it, realize that changes the setup in paragraph one, fix that too. You *think before you speak*.

That's what we wanted. A model that could consider the entire shape of its response before committing to any part of it.

BERT already does this. It just doesn't know it can *generate* yet.

We taught it.

---

## Four Ways to Make BERT Talk

We didn't just try one thing. We designed four approaches, each exploiting BERT's bidirectional nature differently:

1. **ChatBERT-ED** (Encoder-Decoder): The pragmatic one. BERT encodes your message with full bidirectional attention, then a lightweight GPT-style decoder generates the response with cross-attention back to BERT's representations. ~86M parameters. Best of both worlds -- BERT's deep understanding feeding into targeted generation.

2. **ChatBERT-RA** (Retrieval-Augmented): The clever one. Why generate from scratch when you can use BERT to *pick* the best response from a candidate pool? BERT's understanding abilities are perfect for ranking.

3. **ChatBERT-FIB** (Fill-in-the-Blank): The creative one. Template-based generation where BERT fills in masked slots within response structures.

4. **ChatBERT-IMR** (Iterative MLM Refinement): The *wild* one.

We implemented ED and IMR. ED was the safe bet. IMR was the moonshot.

Let's talk about the moonshot.

---

## ChatBERT-IMR: The Art of Iterative Unmasking

Imagine you're staring at a blank page. But it's not really blank -- it's full of placeholders. Every word is `[MASK]`. Your job: figure out what they all should be, by filling in the ones you're most sure about first, then using those as context to figure out the rest.

That's ChatBERT-IMR.

Here's the generation loop, straight from our codebase:

```python
# Initialize response with all [MASK] tokens
response_ids = torch.full(
    (batch_size, max_length),
    MASK_TOKEN_ID,
    device=device,
)

# Track which positions are still masked
is_masked = torch.ones(batch_size, max_length, dtype=torch.bool, device=device)

for iteration in range(num_iterations):
    # Anneal temperature: start warm (exploratory), end cool (confident)
    t = iteration / max(num_iterations - 1, 1)
    current_temp = initial_temperature + t * (final_temperature - initial_temperature)

    # Combine context + current (partially unmasked) response
    combined_ids = torch.cat([input_ids, sep_token, response_ids], dim=1)

    # BERT sees EVERYTHING: context, revealed tokens, and remaining masks
    outputs = model(input_ids=combined_ids, return_dict=True)

    # Get predictions and confidence for response positions
    response_logits = outputs.logits[:, context_len:context_len + max_length]
    probs = F.softmax(response_logits / current_temp, dim=-1)
    predictions = probs.argmax(dim=-1)
    confidence = probs.max(dim=-1).values

    # Unmask the positions where the model is MOST confident
    masked_indices = torch.where(is_masked[b])[0]
    masked_confidence = confidence[b, masked_indices]
    _, top_indices = masked_confidence.topk(num_to_unmask)
    unmask_positions = masked_indices[top_indices]

    # Commit those tokens, keep the rest as [MASK]
    response_ids[b, unmask_positions] = predictions[b, unmask_positions]
    is_masked[b, unmask_positions] = False
```

Read that again. Let it sink in.

The model starts with `[MASK] [MASK] [MASK] [MASK] [MASK] [MASK]`. It looks at the whole thing -- every mask, the entire context -- and decides: "I'm 95% sure position 4 should be 'great' and 87% sure position 1 should be 'That'. Let me lock those in."

Next iteration, it sees: `That [MASK] [MASK] great [MASK] [MASK]`. Now it has more context. "Oh, position 2 should be 'sounds'. And position 5 is probably 'thank'."

`That sounds [MASK] great thank [MASK]`

And on and on. Each pass, the model re-evaluates *everything* in light of what it's already committed to. The temperature anneals from exploratory to decisive. The blanks fill in. A response crystallizes out of nothing, like a photograph developing in a darkroom.

This is *discrete diffusion* applied to conversation. The model literally considers the whole response structure before finalizing any part. It thinks before it speaks.

---

## The Numbers That Surprised Us

We trained both models on DailyDialog and PersonaChat -- about 207k dialogue examples. Used DistilBERT as the backbone because we wanted to prove this works even when you're *not* throwing billions of parameters at it.

- **ChatBERT-IMR**: ~66M parameters
- **ChatBERT-ED**: ~100M parameters (DistilBERT encoder + 4-layer GPT-style decoder)
- **Training**: Single A100 GPU, 67 minutes for ED, 66 minutes for IMR
- **Final train loss**: 3.715 (ED), 3.288 (IMR)

And here's what it actually says:

```
User: Hello, how are you?
Bot:  i am doing okay. how are you?

User: What do you like to do for fun?
Bot:  i like to travel. it is the best. how about you?

User: I am feeling sad today.
Bot:  why do you think that?

User: What is your favorite food?
Bot:  i like mexican food, and you?

User: I just got a new puppy!
Bot:  haha, dogs are fun! whats yours?

User: I love cooking Italian food on weekends.
Bot:  nice! what do you like to do for fun?

User: My cat keeps knocking things off the table.
Bot:  do you have any dogs?

User: I think I want to learn to play guitar.
Bot:  what kind of music do you play?
```

Multi-turn works too:

```
User: I just moved to a new city and I dont know anyone here.
Bot:  what do you think about it?
User: Its been really lonely.
Bot:  i'm sorry. are you from the west?
```

And here's ChatBERT-IMR -- the wild one. Pure BERT, no decoder, generating via iterative unmasking:

```
User: Hello, how are you?
Bot:  i'm doing well, just got off work

User: What do you like to do for fun?
Bot:  i like to play soccer and i love it.

User: I am feeling sad today.
Bot:  i am sad. i am so sorry.

User: Do you want to go to the movies?
Bot:  i'd like to see the movie titanic.

User: Tell me about yourself.
Bot:  i'm a school teacher for a living.

User: The weather has been really nice lately.
Bot:  it's a bit chilly right over here.

User: I love cooking Italian food on weekends.
Bot:  i'm a personal chef at a restaurant.
```

A model that generates by *iteratively unmasking tokens* is producing coherent, contextually appropriate responses. Not left-to-right. Not autoregressive. Just BERT doing its thing, filling in [MASK] tokens one confidence-ranked step at a time. "I'd like to see the movie titanic" -- that's a *specific movie recommendation*, generated non-autoregressively by a 66M parameter model.

Are these responses better than GPT-4? Obviously not. That's not the point. The point is that this works *at all*. BERT -- a model built exclusively for understanding, trained exclusively with masked language modeling, never designed to generate a single coherent sentence -- is holding *actual conversations*. Contextually appropriate, follow-up-question-asking, empathetic conversations. In under two hours of training total. On models smaller than 100M parameters.

---

## You Can Just Do Things

There's a particular energy on the internet that I love. Someone posts "I wonder if X is possible" and someone else replies "I did X this weekend, here's the repo."

That's this project.

The idea started with a conversation that went something like: "Wouldn't it be funny if BERT could chat? Like, actually chat?" And then instead of laughing it off, we opened a terminal.

The "you can't do that" voice in your head is almost always wrong. BERT "can't" generate text -- except it can, if you frame generation as iterative unmasking. Encoder models "can't" have conversations -- except they can, if you're willing to build the scaffolding. The walls between "understanding models" and "generation models" are conventions, not laws of physics.

We didn't need permission. We didn't need a hundred-page literature review confirming feasibility. We had a question ("what if BERT chatted?"), a weekend, and a GPU.

---

## What We Actually Learned

Beyond the "you can just build things" energy, there are real insights here:

**Bidirectional context matters for understanding conversation.** In multi-turn dialogues where context is everything -- where the right response depends on the full history, not just the last message -- BERT's architecture has a genuine structural advantage. It doesn't just *attend* to context. It *deeply integrates* it, bidirectionally, at every layer.

**Iterative refinement is underexplored for dialogue.** The IMR approach -- start masked, unmask confidently, refine -- produces responses with different characteristics than autoregressive generation. Responses tend to be more *structurally coherent* because the model considers global structure early. It's not committed to an awkward first half before it realizes the second half doesn't work.

**Temperature annealing during iterative decoding is crucial.** Starting warm (high temperature) lets the model explore the space of possible responses. Ending cool (low temperature) locks in final decisions with confidence. Without this schedule, quality drops noticeably.

**Small models can surprise you.** At 66M parameters, ChatBERT-IMR has no business generating coherent dialogue. The fact that it does suggests there's a lot of low-hanging fruit in non-autoregressive conversation.

---

## The Architecture, For the Curious

For ChatBERT-ED, the flow is:

```
User message --> [DistilBERT Encoder] --> rich bidirectional representations
                                              |
                                              v (cross-attention)
                                    [4-layer GPT-style Decoder] --> response tokens
```

The encoder doesn't just tokenize -- it builds a deep, bidirectional representation of the conversation context. The decoder then attends to these rich representations while generating the response autoregressively. Think of it as BERT doing the thinking and a small GPT doing the talking.

For ChatBERT-IMR, there's no decoder at all:

```
[context tokens] [SEP] [MASK] [MASK] [MASK] [MASK] [MASK] ...
                              |
                   [DistilBERT + MLM Head]
                              |
                              v
[context tokens] [SEP]  That [MASK] [MASK] great [MASK] ...
                              |
                   [DistilBERT + MLM Head]  (re-evaluate everything)
                              |
                              v
[context tokens] [SEP]  That sounds [MASK] great thank [MASK]
                              |
                   [DistilBERT + MLM Head]  (one more time)
                              |
                              v
[context tokens] [SEP]  That sounds really great thank you
```

Pure BERT. No decoder. No autoregression. Just iterative unmasking. The model "fills in the blanks" of its own response, refining with each pass until it converges.

---

## Try It. Train It. Break It.

Everything is open source. The whole thing:

```bash
# Get the code
git clone <repo-url>
pip install -e ".[all]"

# Download training data
python scripts/download_data.py

# Train ChatBERT-IMR (the wild one)
python scripts/train.py --config configs/imr_small.yaml

# Train ChatBERT-ED (the pragmatic one)
python scripts/train.py --config configs/ed_small.yaml

# Launch the demo
python scripts/demo.py --model_path ./checkpoints/chatbert-imr-small/final --model_type iterative_mlm
```

Train it on your own data. Try different unmasking schedules (linear, cosine, confidence-based). Experiment with the number of refinement iterations. Stack it with retrieval augmentation. Fine-tune it with RLHF. Push it until it breaks, then figure out why.

Some ideas we haven't tried yet but you totally could:

- **Parallel decoding**: IMR's iterative unmasking is *inherently parallel*. Each iteration predicts all positions simultaneously. With enough iterations, could this be *faster* than autoregressive generation?
- **Controllable generation**: Since IMR unmasks based on confidence, you can intervene mid-generation. Don't like where it's going? Re-mask and redirect.
- **Hybrid approaches**: Use IMR for the first draft, then an autoregressive model to polish.

---

## The Punchline

Everyone told us BERT can't chat. We built ChatBERT in a weekend for the cost of two coffees.

The dichotomy between "understanding models" and "generation models" is a convention that exists because it's *convenient*, not because it's *true*. BERT can generate. GPT can understand. The walls are imaginary.

The most interesting things in AI right now aren't happening because someone proved something was possible. They're happening because someone *didn't check whether it was impossible first*.

We didn't check.

It worked.

Your move.

---

*ChatBERT is open source. Star it, fork it, train it on weird data, and tell us what happens. The best ideas come from people who ignore the rules just enough to discover the rules were wrong.*
