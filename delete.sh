echo -n "Delete MNAR (y/n)? "
read answer
if [[ $answer =~ ^[Yy]$ ]]
then
    cd results/openml
    ls | grep pickle | grep MNAR
    ls | grep pickle | grep MNAR | xargs rm  
    cd ../..
fi
echo -n "Delete MCAR (y/n)? "
read answer
if [[ $answer =~ ^[Yy]$ ]]
then
    cd results/openml
    ls | grep pickle | grep MCAR
    ls | grep pickle | grep MCAR | xargs rm
    cd ../..
fi
echo -n "Delete Baseline (y/n)? "
read answer
if [[ $answer =~ ^[Yy]$ ]]
then
    cd results/openml
    ls | grep pickle | grep None
    ls | grep pickle | grep None | xargs rm  
    cd ../..
fi
echo -n "Delete hyperparams (y/n)? "
read answer
if [[ $answer =~ ^[Yy]$ ]]
then
    cd results/openml/hyperparams
    ls | grep trans
    ls | grep trans | xargs rm
    cd ../../..
fi
