#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000
jrc_train_split="1m"

URLS=(
    "http://data.statmt.org/wmt19/translation-task/fr-de/bitexts/europarl-v7.de.gz"
    "http://data.statmt.org/wmt19/translation-task/fr-de/bitexts/europarl-v7.fr.gz"
    "http://data.statmt.org/wmt19/translation-task/fr-de/bitexts/commoncrawl.de.gz"
    "http://data.statmt.org/wmt19/translation-task/fr-de/bitexts/commoncrawl.fr.gz"
    "http://data.statmt.org/wmt19/translation-task/fr-de/bitexts/de-fr.bicleaner07.de.gz"
    "http://data.statmt.org/wmt19/translation-task/fr-de/bitexts/de-fr.bicleaner07.fr.gz"
    "http://data.statmt.org/wmt19/translation-task/fr-de/bitexts/dev08_14.de.gz"
    "http://data.statmt.org/wmt19/translation-task/fr-de/bitexts/dev08_14.fr.gz"
    "https://drive.switch.ch/index.php/s/snHrsUmVizOXuKh/download?path=%2F&files=jrc.train${jrc_train_split}.de.gz"
    "https://drive.switch.ch/index.php/s/snHrsUmVizOXuKh/download?path=%2F&files=jrc.train${jrc_train_split}.fr.gz"
    "https://drive.switch.ch/index.php/s/snHrsUmVizOXuKh/download?path=%2F&files=jrc.test.de.gz"
    "https://drive.switch.ch/index.php/s/snHrsUmVizOXuKh/download?path=%2F&files=jrc.test.fr.gz"
)
FILES=(
    "europarl-v7.de.gz"
    "europarl-v7.fr.gz"
    "commoncrawl.de.gz"
    "commoncrawl.fr.gz"
    "de-fr.bicleaner07.de.gz"
    "de-fr.bicleaner07.fr.gz"
    "dev08_14.de.gz"
    "dev08_14.fr.gz"
    "jrc.train${jrc_train_split}.de.gz"
    "jrc.train${jrc_train_split}.fr.gz"
    "jrc.test.de.gz"
    "jrc.test.fr.gz"
)
CORPORA=(
    "europarl-v7"
    "commoncrawl"
    "de-fr.bicleaner07"
    "dev08_14"
    "jrc.train${jrc_train_split}"
)

OUTDIR=wmt19_jrc_${jrc_train_split}_de_fr

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=fr
lang=de-fr
prep=$OUTDIR
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget -O $file "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -3} == ".gz" ]; then
            gzip -d $file
        fi
    fi

done
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 32 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    cat $orig/jrc.test.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 32 -a -l $l > $tmp/test.$l
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

TRAIN=$tmp/train.de-fr
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done
