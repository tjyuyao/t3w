cd ~/src/t3w/docs
make clean html
cd ~/src/t3w_ghpages
cp -r ~/src/t3w/docs/_build/html/* .
rm -r ~/src/t3w/docs/_build/html/
cd ~/src/t3w/docs/_build/
ln -s ~/src/t3w_ghpages html