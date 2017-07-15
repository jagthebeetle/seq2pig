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

If you're ready, tweak model.py or experiment.py and run:
```
python experiment.py [--batch_size=50] [--input_seq_lens==6] [--logdir=tf_logs]
```

You may optionally replace words.csv with a single-columned list of words. These are assumed to occur one per line and to be taken from the alphabet [a-z] (no uppercase letters). The first line is assumed to be 'words'.

### But why? 🐷
I didn't immediately find a modern yet terse implementation. It's too easy to lose track of what a symbolic graph is doing when the actual equations involved are not that long.

### Other considerations
Currently, only fixed-length sequences are fed into the system, and fixed-length training samples (padded with a null token if needed). The latter issue is a flaw in the model somewhere, but TensorFlow complains if I suggest that training sequences are of varying lengths.
