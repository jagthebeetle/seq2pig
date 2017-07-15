# seq2pig

*A work in progress, in pursuit of the holy grail of machine learning, English to Pig Latin translation.*

Current attempts have been thwarted and yielded results like:

* earaaaaaa
* onggyyyyy
* earaaaaaa
* aslaaaaaa

Truth be told, the above sequence sounds more porcine than Pig Latin, so the ML spirits may just be delivering above and beyond my expectations.

### Details

In all seriousness, this is a mostly correct implementation of Dzimitry Bahdanau et al.'s [RNNSearch model](https://arxiv.org/pdf/1409.0473.pdf), which introduced the alignment / attention mechanism.

The code makes sporadic use of the `@` operator, and was written with Python 3.5.3, on a Windows machine.

The `tensorflow.__version__` is 1.2.1.
