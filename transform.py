from abc import ABCMeta
import copy
from dataclasses import dataclass
from enum import Enum
import inspect
import types
from typing import Callable, Type

def _check_task_transform_signature(f):
    if not callable(f):
        return False

    sig = inspect.signature(f)
    params = sig.parameters.values()
    
    # Calculate the number of positional parameters without defaults, and whether there are args or kwargs present
    num_positional = len([p for p in params if p.default == inspect.Parameter.empty])
    has_args_or_kwargs = any(p.kind in {p.VAR_POSITIONAL, p.VAR_KEYWORD} for p in params)

    # Check if `f` is an unbound method
    is_unbound_method = list(params[0]).name == "self"

    # Number of expected positionals:
    expected_positionals = 3 if is_unbound_method else 2

    return num_positional == expected_positionals and not has_args_or_kwargs

def dispatch_by_type(object_type):
    def _inner(func):
        # check that `func` has the right signature (two params, no args or kwargs)
        assert _check_task_transform_signature(func)
        return _MultiTaskDispatcher(func, object_type)
    return _inner

@dataclass
class _MultiTaskDispatcher:
    func : Callable
    object_type : Type

class _MultiTaskTransformMeta(ABCMeta):
    def __new__(meta_cls, name, bases, dct):
        __dispatch_dict__ = {}
        for key, val in dct.items():
            if isinstance(val, _MultiTaskDispatcher) and val.func.__name__ == "__call__":
                __dispatch_dict__[val.object_type] = val.func

        dct["__dispatch_dict__"] = __dispatch_dict__
        return super().__new__(meta_cls, name, bases, dct)

class HeterogeneousDataError(Exception):
    """Exception thrown by AbstractMultiTaskTransform, depending on the MultiTaskReturnPolicy
    """

class DynamicDispatchError(Exception):
    """Exception thrown by AbstractMultiTaskTransform, when no delegate was found for a data type.
    """
    def __init__(self, cls, input_type):
        super().__init__(
            (f"{cls} transform received a Task/MultiTask object for which no delegate for its data's type could be found. "
            f"The data received has type {input_type}. If this is a data type you'd like to be able to transform, ensure that "
            f"you've specified your delegate correctly:\n"
            "...\n"
            f"@dispatch_by_type({input_type})\n"
            f"def __call__(self, item : {input_type}, shared_parameters : Any):\n"
            "   ...\n")
        )

class MultiTaskReturnPolicy(Enum):
    AlwaysBoth = 0
    """
    Always return two values: the first being the task with transformed labels, the second 
    being the task with untransformed labels. Either value can be None, depending on which
    labels could be transformed.
    """
    OnlyTransformedOrNone = 1
    """
    Only return the transformed labels - None if no labels could be transformed.
    """
    OnlyTransformedStrict = 2
    """
    Only return the transformed labels - raise an exception if any labels couldn't be transformed.
    """
    AllTransformedStrict = 3
    """
    Only return the transformed labels - raise an exception if no labels could be transformed.
    """
    AllBothOrIdentity = 4
    """
    If no labels could be transformed, return a single item: the original, untransformed labels.
    If some labels could be transformed, return two items: the transformed and untransformed labels respectively.
    If all labels could be transformed, return a single item: the transformed labels.
    """
    AllBothOrNone = 5
    """
    If no labels could be transformed, return None.
    If some labels could be transformed, return two items: the transformed and untransformed labels respectively.
    If all labels could be transformed, return a single item: the transformed labels.
    """
    OnlyTransformedOrIdentity = 6
    """
    If no labels could be transformed, return the original, untransformed labels.
    Otherwise, return only the transformed labels.
    """




class AbstractMultiTaskTransform(metaclass = _MultiTaskTransformMeta):

    data_only = False

    def __init__(
        self, 
        multitask_return_policy : MultiTaskReturnPolicy = MultiTaskReturnPolicy.AllBothOrIdentity
    ):
        assert isinstance(multitask_return_policy, MultiTaskReturnPolicy)
        self.multitask_return_policy = multitask_return_policy

    @abstractmethod
    def get_shared_parameters(self, task : MultiTask):
        raise NotImplementedError

    def _get_delegate(self, item):
        return self.__dispatch_dict__.get(type(item), None)

    def _call_multi_task(item : MultiTask) -> Union[None, MultiTask, Tuple[Optional[MultiTask], Optional[MultiTask]]]:
        shared_params = self.get_shared_parameters(item)

        # Try to get the data delegate
        data_delegate = self._get_delegate(item.data)
        if data_delegate is None:
            raise DynamicDispatchError(self.__class__, type(item.data))

        # Transform the data
        transformed_data = data_delegate(item.data, shared_parameters)

        # Check if this transform has been flagged as 'data_only'
        if self.data_only:
            return MultiTask(transformed_data, item.labels)

        # Split the labels in the MultiTask into
        transformed_labels = []
        untransformed_labels = []
        for label in item.labels:
            delegate = self._get_delegate(label)
            if delegate is None:
                untransformed_labels.append(label)
            else:
                transformed_labels.append(delegate(label, shared_params))

        mrp = self.multitask_return_policy
        if not transformed_labels:
            # No labels were transformed. There are four things that can happen here.

            # Return None
            if mrp in {
                MultiTaskReturnPolicy.OnlyTransformedOrNone,
                MultiTaskReturnPolicy.AllBothOrNone
            }:
                return None
            # Return None, and the original untransformed item
            elif mrp == MultiTaskReturnPolicy.AlwaysBoth:
                return None, item
            # Return only the untransformed item
            elif mrp in {
                MultiTaskReturnPolicy.AllBothOrIdentity,
                MultiTaskReturnPolicy.OnlyTransformedOrIdentity
            }:
                return item
            # Raise an exception
            elif mrp in {
                MultiTaskReturnPolicy.OnlyTransformedStrict,
                MultiTaskReturnPolicy.AllTransformedStrict
            }:
                raise HeterogeneousDataError(
                    (f"{self.__class__} transform received a MultiTask object for which none of the labels could be transformed. "
                    f"To handle this exception, try changing this transform's `multitask_return_policy` to something other than the "
                    f"current policy: {mrp.value}.")
                )

        # Combine the transformed data with the transformed labels
        transformed = MultiTask(transformed_data, transformed_labels)

        if untransformed_labels:
            # Some labels couldn't be transformed. There are three things that can happen here.

            # 1. Only return the transformed labels
            if mrp in {
                MultiTaskReturnPolicy.OnlyTransformedOrNone, 
                MultiTaskReturnPolicy.OnlyTransformedStrict, 
                MultiTaskReturnPolicy.OnlyTransformedOrIdentity
            }:
                return transformed
            # 2. Return both the transformed labels and the untransformed labels
            elif mrp in {
                MultiTaskReturnPolicy.AlwaysBoth, 
                MultiTaskReturnPolicy.AllBothOrIdentity, 
                MultiTaskReturnPolicy.AllBothOrNone
            }:
                return transformed, MultiTask(item.data, untransformed_labels)
            # 3. Raise an exception
            elif mrp == MultiTaskReturnPolicy.AllTransformedStrict:
                raise HeterogeneousDataError(
                    (f"{self.__class__} transform received a MultiTask object for which some of the labels could not be transformed. "
                    f"To handle this exception, try changing this transform's `multitask_return_policy` to something other than the "
                    f"current policy: {mrp.value}.")
                )
        
        # Finally, there are two things that can happen here.
        # Either we can return transformed and None, the None standing in for the untransformed labels.
        if mrp == MultiTaskReturnPolicy.AlwaysBoth:
            return transformed, None
        # In any other case, just return the single, transformed item
        return transformed
        

    def _call_single_task(self, item):
        # Use the task type to construct a new task instance of the same type, to avoid
        # modifying the task object in-place.
        _Task = type(task)
        # If item is a Task, apply to both the data and label separately
        data_delegate = self._get_delegate(item.data)
        if data_delegate is None:
            raise DynamicDispatchError(self.__class__, type(item.data))

        shared_parameters = self.get_shared_parameters(item)
        transformed_data = data_delegate(item.data, shared_parameters)
        # If we're a data-only transform, return early
        if self.data_only:
            return _Task(transformed_data, item.label)

        label_delegate = self._get_delegate(item.label)
        if label_delegate is None:
            raise ValueError

        shared_parameters = self.get_shared_parameters(item)
        transformed_data = data_delegate(item.data, shared_parameters)
        transformed_label = label_delegate(item.label, shared_parameters)
        return type(item)(transformed_data, transformed_label) # This is sketchy


    def __call__(self, item):
        if isinstance(item, MultiTask):
            # If item is MultiTask, apply to both the data and labels separately.
            return self._call_multi_task(item)

        elif isinstance(item, Task):
            # If the item is any other Task type, apply to both the data and label separately.
            return self._call_single_task(item)
        
        # Otherwise, apply directly to item
        delegate = self._get_delegate(item)
        if delegate is None:
            raise ValueError
        return delegate(task, self.get_shared_parameters(item))


class AbstractDataOnlyTransform(AbstractMultiTaskTransform):
    data_only = True


@functional_datapipe("transform")
class TransformIterDataPipe(dp.iter.IterDataPipe):
    def __init__(
        self, 
        source_datapipe : dp.iter.IterDataPipe, 
        transform : Union[AbstractMultiTaskTransform, Callable],
        skip_none : Optional[bool] = True,
        flatten : Optional[bool] = True,
        multitask_return_policy : Optional[MultiTaskReturnPolicy] = None
    ):
        self.source_datapipe = source_datapipe
        self.skip_none = skip_none
        self.flatten = flatten

        # Check if a multitask_return_policy was provided, if the transform is an AbstractMultiTaskTransform
        # and whether the transform's return policy differs from what was provided
        if multitask_return_policy is not None and \
            isinstance(transform, AbstractMultiTaskTransform) and \
            transform.multitask_return_policy != multitask_return_policy:
            # If so, make a deepcopy with the new return policy
            transform = copy.deepcopy(transform)
            transform.multitask_return_policy = multitask_return_policy

        self._transform = transform

    def __iter__(self):
        for item in self.source_datapipe:
            outs = self._transform(item)

            # Check if we need to flatten the output
            if self.flatten and isinstance(outs, (list, tuple)):
                # Check for Nones
                if self.skip_none:
                    outs = filter(lambda x: x is None, outs)
                for out in outs:
                    yield out
                continue
            # Check for None
            elif self.skip_none and outs is None:
                continue
            yield outs

    def __len__(self):
        return len(self.source_datapipe)

