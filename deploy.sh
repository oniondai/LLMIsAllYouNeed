#!/bin/sh

model=(
Qwen2.5-1.5B
)

root=`pwd`
for i in ${model[@]};do
  if [ -d "$i" ];then
    continue
  fi
  rm -rf $i;
  mkdir $i;
  cd $i;
  touch README.md;
  echo 'tensorflow==2.13.0' > requirements.txt;
  mkdir src;
  mkdir scripts;
  mkdir data;
  mkdir paper;
  cd src;
  mkdir model;
  touch model/${i}.py
  mkdir utils;
  cd -;
  cd scripts;
  touch run.sh
  echo '#!/bin/sh' > run.sh
  cd -;
  cd ${root};
done

