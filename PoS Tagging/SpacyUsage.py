#!/usr/bin/env python
# coding: utf-8
import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(u"There is an old woman who lives in a hidden place that everyone knows in their souls but few have ever seen.")

for token in doc:
    print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')

POS_counts = doc.count_by(spacy.attrs.POS)
for k,v in sorted(POS_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{5}}: {v}')


DEP_counts = doc.count_by(spacy.attrs.DEP)

for k,v in sorted(DEP_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{14}}: {v}')


from spacy import displacy

doc = nlp(u"Europe was a good place for starting a company to make money at previous years. "
          u"However, there are now 6 or more leading trades like Sony, Apple, etc which are the leaders at products.")
displacy.render(doc, style='ent', jupyter=True)


colors = {'ORG': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)', 'LOC': 'radial-gradient(yellow, green)',
         'DATE': 'radial-gradient(red, green)'}

options = {'ents': ['ORG', 'LOC', 'DATE'], 'colors':colors}

displacy.render(doc, style='ent', jupyter=True, options=options)



