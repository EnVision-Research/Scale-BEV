# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import functools
import numpy as np
from inspect import getfullargspec
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models.utils.transformer import inverse_sigmoid

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None



def array_converter(to_torch=True,
                    apply_to=tuple(),
                    template_arg_name_=None,
                    recover=True):
    """Wrapper function for data-type agnostic processing.

    First converts input arrays to PyTorch tensors or NumPy ndarrays
    for middle calculation, then convert output to original data-type if
    `recover=True`.

    Args:
        to_torch (Bool, optional): Whether convert to PyTorch tensors
            for middle calculation. Defaults to True.
        apply_to (tuple[str], optional): The arguments to which we apply
            data-type conversion. Defaults to an empty tuple.
        template_arg_name_ (str, optional): Argument serving as the template (
            return arrays should have the same dtype and device
            as the template). Defaults to None. If None, we will use the
            first argument in `apply_to` as the template argument.
        recover (Bool, optional): Whether or not recover the wrapped function
            outputs to the `template_arg_name_` type. Defaults to True.

    Raises:
        ValueError: When template_arg_name_ is not among all args, or
            when apply_to contains an arg which is not among all args,
            a ValueError will be raised. When the template argument or
            an argument to convert is a list or tuple, and cannot be
            converted to a NumPy array, a ValueError will be raised.
        TypeError: When the type of the template argument or
                an argument to convert does not belong to the above range,
                or the contents of such an list-or-tuple-type argument
                do not share the same data type, a TypeError is raised.

    Returns:
        (function): wrapped function.

    Example:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Use torch addition for a + b,
        >>> # and convert return values to the type of a
        >>> @array_converter(apply_to=('a', 'b'))
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> a = np.array([1.1])
        >>> b = np.array([2.2])
        >>> simple_add(a, b)
        >>>
        >>> # Use numpy addition for a + b,
        >>> # and convert return values to the type of b
        >>> @array_converter(to_torch=False, apply_to=('a', 'b'),
        >>>                  template_arg_name_='b')
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> simple_add()
        >>>
        >>> # Use torch funcs for floor(a) if flag=True else ceil(a),
        >>> # and return the torch tensor
        >>> @array_converter(apply_to=('a',), recover=False)
        >>> def floor_or_ceil(a, flag=True):
        >>>     return torch.floor(a) if flag else torch.ceil(a)
        >>>
        >>> floor_or_ceil(a, flag=False)
    """

    def array_converter_wrapper(func):
        """Outer wrapper for the function."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Inner wrapper for the arguments."""
            if len(apply_to) == 0:
                return func(*args, **kwargs)

            func_name = func.__name__

            arg_spec = getfullargspec(func)

            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)

            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}

            all_arg_names = arg_names + kwonly_arg_names

            # in case there are args in the form of *args
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []

            # template argument data type is used for all array-like arguments
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_

            if template_arg_name not in all_arg_names:
                raise ValueError(f'{template_arg_name} is not among the '
                                 f'argument list of function {func_name}')

            # inspect apply_to
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(f'{arg_to_apply} is not '
                                     f'an argument of {func_name}')

            new_args = []
            new_kwargs = {}

            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray

            # non-keyword arguments
            for i, arg_value in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(
                        converter.convert(
                            input_array=arg_value, target_type=target_type))
                else:
                    new_args.append(arg_value)

                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value

            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values

            # keyword arguments and non-keyword arguments using default value
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=kwargs[arg_name],
                            target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]

            # add nameless args provided by *args (if exists)
            new_args += nameless_args

            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data,
                                                         tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for k, v in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data

            if recover:
                return recursive_recover(return_values)
            else:
                return return_values

        return new_func

    return array_converter_wrapper


class ArrayConverter:

    SUPPORTED_NON_ARRAY_TYPES = (int, float, np.int8, np.int16, np.int32,
                                 np.int64, np.uint8, np.uint16, np.uint32,
                                 np.uint64, np.float16, np.float32, np.float64)

    def __init__(self, template_array=None):
        if template_array is not None:
            self.set_template(template_array)

    def set_template(self, array):
        """Set template array.

        Args:
            array (tuple | list | int | float | np.ndarray | torch.Tensor):
                Template array.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to
                to a NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range,
                or the contents of a list or tuple do not share the
                same data type, a TypeError is raised.
        """
        self.array_type = type(array)
        self.is_num = False
        self.device = 'cpu'

        if isinstance(array, np.ndarray):
            self.dtype = array.dtype
        elif isinstance(array, torch.Tensor):
            self.dtype = array.dtype
            self.device = array.device
        elif isinstance(array, (list, tuple)):
            try:
                array = np.array(array)
                if array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
                self.dtype = array.dtype
            except (ValueError, TypeError):
                print(f'The following list cannot be converted to'
                      f' a numpy array of supported dtype:\n{array}')
                raise
        elif isinstance(array, self.SUPPORTED_NON_ARRAY_TYPES):
            self.array_type = np.ndarray
            self.is_num = True
            self.dtype = np.dtype(type(array))
        else:
            raise TypeError(f'Template type {self.array_type}'
                            f' is not supported.')

    def convert(self, input_array, target_type=None, target_array=None):
        """Convert input array to target data type.

        Args:
            input_array (tuple | list | np.ndarray |
                torch.Tensor | int | float ):
                Input array. Defaults to None.
            target_type (<class 'np.ndarray'> | <class 'torch.Tensor'>,
                optional):
                Type to which input array is converted. Defaults to None.
            target_array (np.ndarray | torch.Tensor, optional):
                Template array to which input array is converted.
                Defaults to None.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to
                to a NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range,
                or the contents of a list or tuple do not share the
                same data type, a TypeError is raised.
        """
        if isinstance(input_array, (list, tuple)):
            try:
                input_array = np.array(input_array)
                if input_array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
            except (ValueError, TypeError):
                print(f'The input cannot be converted to'
                      f' a single-type numpy array:\n{input_array}')
                raise
        elif isinstance(input_array, self.SUPPORTED_NON_ARRAY_TYPES):
            input_array = np.array(input_array)
        array_type = type(input_array)
        assert target_type is not None or target_array is not None, \
            'must specify a target'
        if target_type is not None:
            assert target_type in (np.ndarray, torch.Tensor), \
                'invalid target type'
            if target_type == array_type:
                return input_array
            elif target_type == np.ndarray:
                # default dtype is float32
                converted_array = input_array.cpu().numpy().astype(np.float32)
            else:
                # default dtype is float32, device is 'cpu'
                converted_array = torch.tensor(
                    input_array, dtype=torch.float32)
        else:
            assert isinstance(target_array, (np.ndarray, torch.Tensor)), \
                'invalid target array type'
            if isinstance(target_array, array_type):
                return input_array
            elif isinstance(target_array, np.ndarray):
                converted_array = input_array.cpu().numpy().astype(
                    target_array.dtype)
            else:
                converted_array = target_array.new_tensor(input_array)
        return converted_array

    def recover(self, input_array):
        assert isinstance(input_array, (np.ndarray, torch.Tensor)), \
            'invalid input array type'
        if isinstance(input_array, self.array_type):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            converted_array = input_array.cpu().numpy().astype(self.dtype)
        else:
            converted_array = torch.tensor(
                input_array, dtype=self.dtype, device=self.device)
        if self.is_num:
            converted_array = converted_array.item()
        return converted_array


@array_converter(apply_to=('points', 'cam2img'))
def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (torch.Tensor): 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img (torch.Tensor): Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].

    Returns:
        torch.Tensor: points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
         # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes, 
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
      
        # weighted sum of above two costs
        cost = cls_cost + reg_cost
        
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)