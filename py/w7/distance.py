class Distance():
    def __init__(self, object_a=None, object_b=None):
        self._object_a = object_a
        self._object_b = object_b
        self._distance = None
        self._validate_objects()
        self._len = len(self._object_a)

    def _validate_objects(self):
        object_a_type = type(self._object_a)
        object_b_type = type(self._object_b)
        if object_a_type != object_b_type:
            raise Exception(
                "Objects must be of same type to have distance")
        if not (hasattr(self._object_a, '__len__') and hasattr(self._object_b, '__len__')):
            raise Exception(
                "Objects must have __len__ attribute")
        len_object_a = len(self._object_a)
        len_object_b = len(self._object_b)
        if len_object_a != len_object_b:
            raise Exception(
                "Objects must have same length")

    def get_objects(self, verbose="silent"):
        objects_tuple = (
            self._object_a, self._object_b)
        if verbose == "debug":
            print("\nObject tuple -> \n\n",
                  objects_tuple, "\n")
        return objects_tuple

    def get_len(self, verbose="silent"):
        if verbose == "debug":
            print("\nObject lengths -> ",
                  self._len, "\n")
        return self._len

    def get_distance(self, verbose="silent"):
        pass
