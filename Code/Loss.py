# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Classes_Path);

import  numpy;
import  torch;
import  math;
from    typing                  import Tuple, List, Dict;

from    Derivative              import Derivative;
from    Term                    import Term;
from    Network                 import Network, Rational;
from    Evaluate_Derivatives    import Derivative_From_Derivative;


def Data_Loss(
        U           : Network,
        Inputs      : torch.Tensor,
        Targets     : torch.Tensor) -> torch.Tensor:
    """ 
    This function evaluates the data loss, which is the mean square error 
    between U at the Inputs, and the Targets. To do this, we first evaluate U 
    at the data points. At each point (t, x), we then  evaluate 
    |U(t, x) - U'_{t, x}|2, where U'_{t,x} is the data point corresponding to 
    (t, x). We sum up these values and then divide by the number of data 
    points.

    ----------------------------------------------------------------------------
    Arguments:

    U: The neural network which approximates the system response function.

    Inputs: If U is a function of one spatial variable, then this should
    be a two column tensor whose ith row holds the (t, x) coordinate of the
    ith data point. If U is a function of two spatial variables, then this
    should be a three column tensor whose ith row holds the (t, x, y)
    coordinates of the ith data point.

    Targets: If Targets has N rows, then this should be an N element
    tensor whose ith element holds the value of U at the ith data point.

    ----------------------------------------------------------------------------
    Returns:

    A scalar tensor whose sole entry holds the mean square data loss. 
    """

    # Evaluate U at the data points.
    U_Predict = U(Inputs).squeeze();

    # Evaluate the point-wise square difference of U_Predict and Targets.
    Square_Error = ( U_Predict - Targets ) ** 2;

    # Return the mean square error.
    return Square_Error.mean();



def Coll_Loss(
        U           : Network,
        Xi          : torch.Tensor,
        Mask        : torch.Tensor,
        Coll_Points : torch.Tensor,
        Derivatives : List[Derivative],
        LHS_Term    : Term,
        RHS_Terms   : List[Term],
        Device      : torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
    Let L(U) denote the library matrix (i,j entry is the jth RHS term evaluated
    at the ith collocation point. Further, let b(U) denote the vector whose ith 
    entry is the LHS Term at the ith collocation point. Then this function 
    returns ||b(U) - L(U)Xi||_2

    ----------------------------------------------------------------------------
    Arguments:

    U: The Neural Network that approximates the solution.

    Xi: A trainable (requires_grad = True) torch 1D tensor. If there are N
    RHS_Terms, this should be an N element vector.

    Mask: A boolean tensor whose shape matches that of Xi. When adding the kth 
    RHS term to the Library_Xi product, we check if Mask[k] == False. If so, 
    We add 0*Xi[k]. Otherwise, we compute the kth library term as usual.

    Coll_Points: B by n column tensor, where B is the number of coordinates and
    n is the dimension of the problem domain. The ith row of Coll_Points should
    hold the components of the ith collocation points.

    Derivatives: We try to learn a PDE of the form
            T_0(U) = Xi_1*T_1(U) + ... + Xi_n*T_N(U).
    where each T_k is a "Term" of the form
            T_k(U) = (D_1 U)^{p(1)} ... (D_m U)^{p(m)}
    where each D_j is a derivative operator and p(j) >= 1. Derivatives is a list
    that contains each D_j's in each term in the equation above. This list
    should be ordered according to the Derivatives' orders (see Derivative
    class).

    LHS_Term : A Term object representing T_0 in the equation above.

    RHS_Terms : A list of Term objects whose ith entry represents T_i in the
    equation above.

    Device: The device (gpu or cpu) that we train on.

    ----------------------------------------------------------------------------
    Returns:

    A tuple. The first entry of the tuple is a scalar tensor whose lone element
    contains the mean square collocation loss at the Coll_Points. The second is a
    1D tensor whose ith entry holds the PDE residual at the ith collocation
    point. You can safely discard the second return variable if you just want
    to get the loss. 
    """

    # Make sure Xi's length matches RHS_Terms'.
    assert(torch.numel(Xi) == len(RHS_Terms));

    # Make sure Coll_Points requires grad.
    Coll_Points.requires_grad_(True);

    # First, evaluate U at the Coll_Points.
    U_Coords : torch.Tensor = U(Coll_Points).view(-1);



    ############################################################################
    # Form a dictionary housing D_j U, for each derivative D_j in Derivatives.

    # Initialize
    D_U_Dict : Dict[Derivative] = {};

    # Cycle through the derivatives.
    for j in range(len(Derivatives)):
        # Fetch D_j.
        D_j : Derivative = Derivatives[j];

        # Check if we can compute D_j from any of the derivatives of U that
        # we've already computed.
        for i in reversed(range(0, j)):
            # Get the ith derivative.
            D_i : Derivative = Derivatives[i];

            # Check if D_i is a child of D_j.
            if(D_i.Is_Child_Of(D_j)):
                # If so, compute D_j U from D_i U.
                D_j_U : torch.Tensor = Derivative_From_Derivative(
                                        Da      = D_j,
                                        Db      = D_i,
                                        Db_U    = D_U_Dict[tuple(D_i.Encoding)],
                                        Coords  = Coll_Points).view(-1);


                # Store the result in the dictionary.
                D_U_Dict[tuple(D_j.Encoding)] = D_j_U;

                # Break!
                break;
        else:
            # This runs if we do not encounter break in the for loop above.
            # If we end up here, then we can not calculate D_j U from a
            # derivative that we already computed. In this case, we must compute
            # D_j U from U_Coords.
            I : Derivative = Derivative(Encoding = numpy.array([0, 0]));

            # Compute D_j U.
            D_j_U : torch.Tensor = Derivative_From_Derivative(
                                    Da      = D_j,
                                    Db      = I,
                                    Db_U    = U_Coords,
                                    Coords  = Coll_Points).view(-1);

            # Store the result in the dictionary.
            D_U_Dict[tuple(D_j.Encoding)] = D_j_U;



    ############################################################################
    # Construct b(U) (see doc string).

    # Initialize b_U to a vector of all ones.
    b_U : torch.Tensor = torch.ones_like(U_Coords);

    # Cycle through the sub-terms of the LHS Term.
    for i in range(LHS_Term.Num_Sub_Terms):
        # First, fetch the sub-term's derivative, power.
        Di : Derivative = LHS_Term.Derivatives[i];
        pi : int        = LHS_Term.Powers[i];

        # Next, fetch its value from the dictionary.
        Di_U : torch.Tensor = D_U_Dict[tuple(Di.Encoding)];

        # Raise it to the sub term's power.
        Di_U_pi : torch.Tensor = torch.pow(Di_U, pi);

        # Accumulate D_i_U into b_U.
        b_U = torch.multiply(b_U, Di_U_pi);



    ############################################################################
    # Construct L(U)*Xi (see doc string).

    L_U_Xi : torch.Tensor = torch.zeros_like(b_U);

    # Cycle through the RHS Terms.
    for j in range(len(RHS_Terms)):
        # Check if the jth term is masked. If so, move on.
        if(Mask[j] == True):
            L_U_Xi += 0.0*Xi[j];
            continue;
    
        # Get the term.
        T_j     : Term          = RHS_Terms[j];

        # Initialize a tensor to hold T_j(U).
        T_j_U   : torch.Tensor  = torch.ones_like(U_Coords);

        # Cycle through T_j's sub-terms.
        for i in range(T_j.Num_Sub_Terms):
            # First, fetch the derivative, power for T_j's ith sub-term.
            Di : Derivative = T_j.Derivatives[i];
            pi : int        = T_j.Powers[i];

            # Next, fetch its value from the dictionary.
            Di_U : torch.Tensor = D_U_Dict[tuple(Di.Encoding)];

            # Raise it to the sub term's power.
            Di_U_pi : torch.Tensor = torch.pow(Di_U, pi);

            # Accumulate D_i_U into T_j_U.
            T_j_U = torch.multiply(T_j_U, Di_U_pi);

        # Accumulate T_j_U*Xi[j] into L_U_Xi.
        L_U_Xi += torch.multiply(T_j_U, Xi[j]);

    # Now, compute the residual, b(U) - L(U)Xi.
    Residual : torch.Tensor = torch.subtract(b_U, L_U_Xi);

    # Return the mean square residual (Collocation Loss), and the residual.
    return ((Residual**2).mean(), Residual);



def Lp_Loss(Xi : torch.Tensor, Mask : torch.Tensor, p : float):
    """ 
    This function approximates the L0 norm of Xi using the following quantity:
        w_1*|Xi[1]|^2 + w_2*|Xi[2]|^2 + ... + w_N*|Xi[N]|^2
    Where, for each k,
        w_k =   1/max{delta, |Xi[k]|^{p - 2}}       if Mask[k] == False
                0                                   if Mask[k] == True
    (where delta is some small number that ensures we're not dividing by zero!)

    ----------------------------------------------------------------------------
    Arguments:

    Xi: The Xi vector in our setup. This should be a one-dimensional tensor.

    Mask: A boolean tensor whose shape matches that of Xi. When adding the kth 
    RHS term to the Library_Xi product, we check if Mask[k] == False. If so, 
    We add 0*Xi[k]. Otherwise, we compute the kth library term as usual.

    p: The "p" in in the expression above

    ----------------------------------------------------------------------------
    Returns:

        w_1*|Xi[1]|^p + w_2*|Xi[2]|^p + ... + w_N*|Xi[N]|^p
    where N is the number of components of Xi. 
    """

    assert(p > 0 and p < 2)

    # First, square the components of Xi. Also, make a double precision copy of
    # Xi that is detached from Xi's graph.
    delta : float = .0000001;
    Xi_2          = torch.mul(Xi, Xi);
    Xi_Detach     = torch.detach(Xi);

    # Now, define a weights tensor.
    W               = torch.empty_like(Xi_Detach);
    N : int         = W.numel();
    for k in range(N):
        if(Mask[k] == True):
            W[k] = 0.0;
            continue;

        # First, obtain the absolute value of the kth component of Xi, as a float.
        Abs_Xi_k    : float = abs(Xi[k].item());

        # Now, evaluate W[k].
        W_k  = 1./max(delta, Abs_Xi_k**(2 - p));

        # Check for infinity (which can happen, unfortunately, if delta is too
        # small). If so, remedy it.
        if(math.isinf(W_k)):
            print("W_k got to infinity");
            print("Abs_Xi_k = %f" % Abs_Xi_k);
            W_k = 0;

        W[k] = W_k;

    # Finally, evaluate the element-wise product of Xi and W[k].
    W_Xi_2 = torch.mul(W, Xi_2);
    return W_Xi_2.sum();



def L2_Squared_Loss(U : Network) -> torch.Tensor:
    """
    This function computes the square of the L2 norm of U's parameter vector.
    In particular, if W_1, ... , W_D and b_1, ... , b_D are the weight matrices
    and bias vectors of U, respectively, then this function returns a single 
    element tensor whose lone entry holds the value
            ||W_1||_F^2 + ... + ||W_D||_F^2 + ||b_1||_2^2 + ... + ||b_D||_2^2
    (where ||M||_F denotes the Frobenius norm of the matrix A; that is, the sum
    of the squares of the elements of F). 

    Note: If U is a rational NN, we also include the square of the coefficients 
    of each rational activation function.

    ---------------------------------------------------------------------------
    Arguments:

    U: A neural network object.

    ---------------------------------------------------------------------------
    Returns:

    A single element tensor whose lone element holds the square of the L2 norm 
    of U's parameter vector.
    """

    # Cycle through U's layers, adding on the square of the parameters in the 
    # ith layer in the ith loop iteration.

    Loss : torch.Tensor = torch.zeros(1, dtype = torch.float32);

    Num_Layers : int = U.Num_Layers;
    for i in range(Num_Layers):
        # Add on the contribution due to the weight matrix and bias vector.
        W : torch.Tensor = U.Layers[i].weight;
        b : torch.Tensor = U.Layers[i].bias;

        Loss += torch.sum(torch.multiply(W, W));
        Loss += torch.sum(torch.multiply(b, b));

        # if this layer uses a rational NN, we need to add its coefficients.
        AF : torch.nn.Module = U.Activation_Functions[i];
        if(isinstance(AF, Rational)):
            Loss += torch.sum(torch.multiply(AF.a, AF.a));
            Loss += torch.sum(torch.multiply(AF.b, AF.b));
    
    return Loss;
