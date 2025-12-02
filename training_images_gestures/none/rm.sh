for i in {99..150}
do
        rem=$(($i % 2))
        if [ "$rem" -ne "0" ]; then
                echo $i
                rm *0000$i*
        fi
done
