for name in "$@"
do
  python3 test_validation.py --weight_path log/ --model_name "teacher.ckpt-$name" --is_teacher
done

