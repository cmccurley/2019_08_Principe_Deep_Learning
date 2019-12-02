function [NMSE,growth] = QKLMS_MEE(input,output,validateInput,validateDesired,modelOrder,D,eta,BW,errorThresh,lengthTrain,plotFigs,errorOrder,errorBW)
    
    %This function runs a Kernel Least Mean Square adaptive filter
    %on input data give a desired (output) using a Gaussian kernel
    %validateInput = filter input data for validation set
    %validateDesired = desired vector for validation data
    %modelOrder =  filter length
    %D = embedding dimension
    %eta = learning step size
    %BW = bandwidth of Gaussian Kernel
    %errorThresh = threshold to add new kernel centers
    %lengthTrain = num samples to train filter
    %plotFigs = boolean to plot weight tracks/learning curves/error
    
    %Algorithm from Pg. 34 Kernel Adaptive Filters, Principe
    

    %Subtract mean of input data
    input = input - mean(input);
    output = output - mean(input);

    dataSize = length(input);
    
    %Find input power
    inputPower = input'*input;
    valPower = validateInput'*validateInput;
    
    % Popultate data  matrix and desired  vector given a model order
    XTrain  = zeros(dataSize,modelOrder);
    dTrain = zeros(dataSize,1);
    XVal  = zeros(length(validateInput),modelOrder);
    
    %Training
    for i = 1:dataSize
        if i < D*modelOrder-1
            XTrain(i,:) = zeros(1,modelOrder);
        else
            XTrain(i,:) = input(i:-D:i-(modelOrder*D)+1);  
        end
    end
    
    %Desired
    for j  = 1:dataSize
        dTrain(j) = output(j);
    end
   
    %Validation
     for i = 1:length(validateInput)
        if i < D*modelOrder-1
            XVal(i,:) = zeros(1,modelOrder);
        else
            XVal(i,:) = validateInput(i:-D:i-(modelOrder*D)+1);  
        end
    end
   

   %======================================================================
   % Initialize variables
   %======================================================================
   NMSE = zeros(lengthTrain,1);
   MSE = zeros(lengthTrain,1);
   f = zeros(lengthTrain,1);
   error = zeros(lengthTrain,1);
   U = XTrain;
   a =  [];
   growth = zeros(lengthTrain,1);
   
    %======================================================================
    % 
    %======================================================================        
   
    for k = 1:lengthTrain

        if k == 1
            %initialize coefficient vector
            a(k) = eta*dTrain(k);
    
            %Choose initial center
            C = U(k,:)';
            
            %Evaluate estimated output
            f(k) = a(k)*exp(-(1/(2*(BW^2)))*norm(U(k,:)'-U(k,:)'));
            
            %Initial error
            error(k) = dTrain(k) - f(k);
           
        else
%             if mod(k,200) == true
%                disp(['QKLMS iteration: ' num2str(k-1) ' of ' num2str(lengthTrain)]); 
%             end
            
            %calculate estimated output
            for m = 1:size(C,2)
                f(k) = f(k) + a(m)*exp(-(1/(2*(BW^2)))*norm(U(k,:)'-C(:,m)));
            end
            
            %Instantaneous error
            error(k) = dTrain(k) - f(k);
            
            %Update kernel center dictionary
            if k > modelOrder %handle initial zeros in training data
                distances  =  zeros(size(C,2),1);
                
                %calculate distances between existing dictionary and new
                %sample
                for p = 1:size(C,2)
                   distances(p) = norm(U(k,:)'-C(:,p)); 
                end

                %Get smallest distance between current dictionary and new
                %sample
                [dis,~] = min(distances); 
                
                %If distance is  greater than error threshold, add to
                %dictionary
                if dis <= errorThresh
                    if length(a)<=errorOrder %case j is in between
                        for p  = 0:length(a)-1
                            a(end-p) = a(end-p)-((error(k)-error(k-p))/(errorBW^2))*exp(-(1/(2*(errorBW^2)))*(error(k)-error(k-p).^2));
                        end
                    else
                        for p = 0:errorOrder-1
                           a(end-p) = a(end-p)-((error(k)-error(k-p))/(errorBW^2))*exp(-(1/(2*(errorBW^2)))*(error(k)-error(k-p).^2));
                        end
                    end 
                else 
                    %Find new a value to append for case i = j in paper
                    %and update previous a's
                    kernelSum =  0;
                    if k == 1
                        
                    elseif k <=errorOrder
                        for p = 1:k-1
                            kernelSum = kernelSum + ((error(k)-error(p))/(errorBW^2))*exp(-(1/(2*(errorBW^2)))*(error(k)-error(p).^2));
                        end
                    else
                        for p = k-errorOrder+1:k-1
                            kernelSum = kernelSum + ((error(k)-error(p))/(errorBW^2))*exp(-(1/(2*(errorBW^2)))*(error(k)-error(p).^2));
                        end
                    end
                    newA = (eta/errorOrder)*kernelSum;
                    
                    C = horzcat(C,U(k,:)');
                    a = vertcat(a,newA);
                    
                    %update previous a's
                    if length(a)<=errorOrder %case j is in between
                        for p  = 1:length(a)-1
                            a(end-p) = a(end-p)-((error(k)-error(k-p))/(errorBW^2))*exp(-(1/(2*(errorBW^2)))*(error(k)-error(k-p).^2));
                        end
                    else
                        for p = 1:errorOrder
                           a(end-p) = a(end-p)-((error(k)-error(k-p))/(errorBW^2))*exp(-(1/(2*(errorBW^2)))*(error(k)-error(k-p).^2));
                        end
                    end
                end
            end
      
        end
        
        estimated = zeros(length(validateInput),1);
        
        %Calculate validation NMSE and MSE
        for m = 1:length(validateInput)
            for n = 1:size(C,2)
                estimated(m) = estimated(m) + a(n)*exp(-(1/(2*(BW^2)))*norm(C(:,n)-XVal(m,:)'));
            end   
        end
        
        totalError = validateDesired - estimated; 
        NMSE(k) = mean(totalError.^2)/valPower;
        MSE(k) = mean(totalError.^2);      
        
        growth(k) =  size(C,2);

%         if k == lengthTrain
%         keyboard;
%         end
    end
    
    
    
    %======================================================================
    %Plot weight tracks and learning curves 
    %======================================================================
    if plotFigs == true
    
        %plot training curves
         figure;
         plot(1:lengthTrain,NMSE(:,1));
         title(['Learning Curve for Model order of ' num2str(modelOrder) ' and Step Size of ' num2str(eta)]);
         xlabel('Iteration');
         ylabel('NMSE');
         
         figure();
         plot(1:lengthTrain,growth);
         title('Growth Curve');
         xlabel('Iteration');
         ylabel('Network Size');
    end  
    
  end
