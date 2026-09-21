# LaCoSI
Official implementation of LaCoSI. This repository currently provides a partial release of the code. The complete implementation will be made publicly available upon acceptance of the paper.


## Network Architecture
The above dimensions correspond to our implementation. LaCoSI mainly specifies the coordination and information-incorporation mechanism rather than a fixed network architecture, and the network sizes can be adapted to the requirements of different tasks.

LaCoSI keeps the original agent backbone, latent coordination learner, message encoder, individual Q network, and QMIX mixer, and only introduces a lightweight CVoI estimator. Each agent encodes its 204-dimensional observation into a 64-dimensional feature, followed by a GRUCell with hidden size 64. The latent coordination module uses a teacher--student architecture: the view network is \(204 \rightarrow 128 \rightarrow 64\), followed by a projection network \(64 \rightarrow 64 \rightarrow 16\) that outputs the latent coordination distribution. The inferred coordination category is mapped to a 4-dimensional embedding and further encoded by an MLP \(4 \rightarrow 64 \rightarrow 64\).

Teammate messages are generated from the 64-dimensional recurrent hidden state using a \(64 \rightarrow 16\) linear layer with Tanh, and messages from other available agents are mean-pooled into a 16-dimensional candidate representation \(\tilde m_i^t\). The CVoI estimator takes the two-dimensional latent coordination statistics \(\xi_i^t=[u_i^t,v_i^t]\) and uses an MLP \(2 \rightarrow 16 \rightarrow 1\) to predict the scalar information value \(\hat d_i^t\), which determines the receiver-side gate.

For action-value estimation, the 64-dimensional recurrent feature, 64-dimensional coordination feature, and 16-dimensional gated teammate representation are concatenated into a 144-dimensional input, which is mapped to 18 action values by the individual Q head. The target Q head has the same architecture and is reused in the counterfactual branch under information-present and information-absent message inputs. The centralized value function follows QMIX with hidden size 64; the target mixer evaluates the corresponding individual-value vectors together with the global state to obtain \(V_+^t\) and \(V_{-i}^t\), from which \(d_i^t=[V_+^t-V_{-i}^t]_+\) is constructed. Thus, the only additional learnable module introduced by the new mechanism is the CVoI estimator \(2 \rightarrow 16 \rightarrow 1\); the counterfactual branch reuses the existing target Q and target mixing networks.


### Optimization

The three objectives optimize distinct components of LaCoSI. \(\mathcal{L}_{\mathrm{CB}}\) updates the student latent coordination network, \(\mathcal{L}_{\mathrm{CVoI}}\) updates only the information-value estimator \(F_\phi\), and \(\mathcal{L}_{\mathrm{RL}}\) optimizes the message encoder, individual action-value network, and mixing network. The teacher coordination network is updated by exponential moving average, while the target action-value and mixing networks are updated using the standard target-network update.


### Counterfactual Value Normalization

The raw counterfactual value
\[
d_i^t=[V_+^t-V_{-i}^t]_+
\]
is non-negative but unbounded. We normalize it to \([0,1]\) using a
running 95th-percentile scale \(s_d\):
\[
\tilde d_i^t=
\operatorname{clip}
\left(
\frac{d_i^t}{s_d+\epsilon},0,1
\right).
\]
The estimator predicts \(\tilde d_i^t\) with a sigmoid output, and the
receiver-side gate is computed by thresholding the normalized prediction.
