classdef FastAP < dagnn.Loss
    properties
        opt
    end
    
    properties (Transient)
        Z
        Delta
        dist2
        I_pos
        I_neg
        h_pos
        h_neg
        H_pos
        N_pos
        h
        H
    end

    methods

        function obj = FastAP(varargin)
            obj.load(varargin{:});
        end


        function outputs = forward(obj, inputs, params)
            % forward pass
            X = squeeze(inputs{1});  % features (L2 normalized)
            Y = inputs{2};
            N = size(X, 2); 
            assert(size(Y, 1) == N); 

            opts  = obj.opt;
            onGPU = numel(opts.gpus) > 0;

            % binary affinity
            Affinity = 2 * bsxfun(@eq, Y(:,1), Y(:,1)') - 1;
            Affinity(logical(eye(N))) = 0;
            I_pos = (Affinity > 0); 
            I_neg = (Affinity < 0);
            N_pos = sum(I_pos, 2);

            % (squared) pairwise distance matrix
            dist2 = max(0,  2 - 2 * X' * X);

            % histogram binning
            Delta = 4 / opts.nbins;
            Z     = linspace(0, 4, opts.nbins+1);
            L     = length(Z); 
            h_pos = zeros(N, L);
            h_neg = zeros(N, L);
            if onGPU 
                h_pos = gpuArray(h_pos); 
                h_neg = gpuArray(h_neg); 
            end

            for l = 1:L
                pulse = obj.softBinning(dist2, Z(l), Delta);
                h_pos(:, l) = sum(pulse .* I_pos, 2);
                h_neg(:, l) = sum(pulse .* N_neg, 2);
            end
            H_pos = cumsum(h_pos, 2);
            h     = h_pos + h_neg;
            H     = cumsum(h, 2);

            % compute FastAP
            FastAP = h_pos .* H_pos ./ H; 
            FastAP(isnan(FastAP)|isinf(FastAP)) = 0;
            FastAP = sum(FastAP, 2) ./ N_pos;
            FastAP = FastAP(~isnan(FastAP));

            obj.numAveraged = N;
            obj.average = gather(mean(FastAP));

            % output
            outputs{1} = sum(FastAP);
            obj.Z      = Z;
            obj.Delta  = Delta;
            obj.dist2  = dist2;
            obj.I_pos  = I_pos;
            obj.I_neg  = I_neg;
            obj.h_pos  = h_pos;
            obj.h_neg  = h_neg;
            obj.H_pos  = H_pos;
            obj.N_pos  = N_pos;
            obj.h      = h;
            obj.H      = H;
        end


        function [dInputs, dParams] = backward(obj, inputs, params, dOutputs)
            % backward pass
            X     = squeeze(inputs{1});
            opts  = obj.opt;
            onGPU = numel(opts.gpus) > 0;

            L     = numel(obj.Z);
            h_pos = obj.h_pos; 
            h_neg = obj.h_neg; 
            H_pos = obj.H_pos; 
            N_pos = obj.N_pos; 
            H_neg = H - H_pos;
            h     = obj.h; 
            H     = obj.H; 
            H2    = H .^ 2;

            % 1. d(FastAP)/d(h+)
            tmp1 = h_pos .* H_neg ./ H2;
            tmp1(isnan(tmp1)) = 0;

            d_AP_h_pos = (H_pos .* H + h_pos .* H_neg) ./ H2;
            d_AP_h_pos = d_AP_h_pos + tmp1 * triu(ones(L), 1)'; 
            d_AP_h_pos = bsxfun(@rdivide, d_AP_h_pos, N_pos);
            d_AP_h_pos(isnan(d_AP_h_pos)|isinf(d_AP_h_pos)) = 0;

            % 2. d(FastAP)/d(h-)
            tmp2 = -h_pos .* H_pos ./ H2;
            tmp2(isnan(tmp2)) = 0;

            d_AP_h_neg = tmp2 * triu(ones(L))';
            d_AP_h_neg = bsxfun(@rdivide, d_AP_h_neg, N_pos);
            d_AP_h_neg(isnan(d_AP_h_neg)|isinf(d_AP_h_neg)) = 0;

            % 3. d(FastAP)/d(x)
            d_AP_x = 0;
            for l = 1:L
                % NxN matrix of delta_hat(i, j, l) for fixed l
                dpulse = obj.dSoftBinning(obj.dist2, obj.Z(l), obj.Delta);
                dpulse(isnan(dpulse)|isinf(dpulse)) = 0;
                ddp = dpulse .* obj.I_pos;
                ddn = dpulse .* obj.I_neg;

                alpha_p = diag(d_AP_h_pos(:, l));  % NxN
                alpha_n = diag(d_AP_h_neg(:, l));  
                Ap = ddp * alpha_p + alpha_p * ddp;
                An = ddn * alpha_n + alpha_n * ddn;

                % accumulate gradient % (BxN) (NxN) -> (BxN)
                d_AP_x = d_AP_x - X * (Ap + An);
            end

            % output
            dInputs{1} = zeros(size(inputs{1}), 'single');
            if onGPU, dInputs{1} = gpuArray(dInputs{1}); end
            dInputs{1}(1, 1, :, :) = -single(d_AP_x);
            dInputs{2} = [];
            dParams = {};
        end


        function [grad, objval] = computeGrad(obj, X, Y)
            % helper function to compute gradient matrix
            inputs = {X, Y};
            output = obj.forward(inputs, []);
            objval = obj.average;
            [dInputs, ~] = obj.backward(inputs, [], {'objective', 1});
            grad = dInputs{1};
        end


        function y = softBinning(D, mid, delta)
            %     D: input matrix of distance values
            %   mid: scalar, the center of some histogram bin
            % delta: scalar, histogram bin width
            %
            % For histogram bin mid, compute the contribution y
            % from every element in D.  
            y = 1 - abs(D - mid) / delta;
            y = max(0, y);
        end


        function y = dSoftBinning(D, mid, delta);
            % vectorized version
            % mid: scalar bin center
            %   D: can be a matrix
            ind1 = (D > mid-delta) & (D <= mid);
            ind2 = (D > mid) & (D <= mid+delta);
            y = (ind1 - ind2) / delta;
        end

    end
end
