def get_inception_renet_v2_output_size(input_size):

    size = valid_padding_non_unit_strides(input_size, 3, 2)

    size = valid_padding_non_unit_strides(size, 3)

    size = valid_padding_non_unit_strides(size, 3, 2)

    size = valid_padding_non_unit_strides(size, 3)

    size = valid_padding_non_unit_strides(size, 3, 2)

    size = valid_padding_non_unit_strides(size, 3, 2)

    size = valid_padding_non_unit_strides(size, 3, 2)

    return size


def reverse_inception_renet_v2(output):

    ret = rev_valid_padding_non_unit_strides(output, 3, 2)
    ret = rev_valid_padding_non_unit_strides(ret, 3, 2)
    ret = rev_valid_padding_non_unit_strides(ret, 3, 2)
    ret = rev_valid_padding_non_unit_strides(ret, 3, 1)
    ret = rev_valid_padding_non_unit_strides(ret, 3, 2)
    ret = rev_valid_padding_non_unit_strides(ret, 3, 1)
    ret = rev_valid_padding_non_unit_strides(ret, 3, 2)
    return ret


def valid_padding_non_unit_strides(input_size, kernal_size, strides = 1):
    return (input_size - kernal_size)/strides + 1

def rev_valid_padding_non_unit_strides(out_put, kernal_size, strides = 1):
    return (out_put - 1) * strides + kernal_size


def same_padding_non_unit_strides(input_size, kernal_size, strides = 1):
    return (input_size + 2 * (kernal_size / 2) - kernal_size) / strides + 1


print get_inception_renet_v2_output_size(651)

print reverse_inception_renet_v2(10)