local Module = torch.class('nn.Module')

function Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
end

function Module:parameters()
   if self.weight and self.bias then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
   elseif self.weight then
      return {self.weight}, {self.gradWeight}
   elseif self.bias then
      return {self.bias}, {self.gradBias}
   else
      return
   end
end

function Module:updateOutput(input)
   return self.output
end

function Module:forward(input)
   return self:updateOutput(input)
end

function Module:backward(input, gradOutput)
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput)
   return self.gradInput
end

function Module:backwardUpdate(input, gradOutput, lr)
   self:updateGradInput(input, gradOutput)
   self:accUpdateGradParameters(input, gradOutput, lr)
   return self.gradInput
end

function Module:updateGradInput(input, gradOutput)
   return self.gradInput
end

function Module:accGradParameters(input, gradOutput, scale)
end

function Module:accUpdateGradParameters(input, gradOutput, lr)
   local gradWeight = self.gradWeight
   local gradBias = self.gradBias
   self.gradWeight = self.weight
   self.gradBias = self.bias
   self:accGradParameters(input, gradOutput, -lr)
   self.gradWeight = gradWeight
   self.gradBias = gradBias
end

function Module:sharedAccUpdateGradParameters(input, gradOutput, lr)
   if self:parameters() then
      self:zeroGradParameters()
      self:accGradParameters(input, gradOutput, 1)
      self:updateParameters(lr)
   end
end

function Module:zeroGradParameters()
   local _,gradParams = self:parameters()
   if gradParams then
      for i=1,#gradParams do
         gradParams[i]:zero()
      end
   end
end

function Module:updateParameters(learningRate)
   local params, gradParams = self:parameters()
   if params then
      for i=1,#params do
         params[i]:add(-learningRate, gradParams[i])
      end
   end
end

function Module:share(mlp, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(mlp[v])
         self.accUpdateGradParameters = self.sharedAccUpdateGradParameters
         mlp.accUpdateGradParameters = mlp.sharedAccUpdateGradParameters
      end
   end
   return self      
end

function Module:clone(...)
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   if select('#',...) > 0 then
      clone:share(self,...)
   end
   return clone
end

function Module:type(type)
   -- find all tensors and convert them
   for key,param in pairs(self) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self[key] = param:type(type)
      end
   end
   -- find submodules in classic containers 'modules'
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:type(type)
      end
   end
   return self
end

function Module:float()
   return self:type('torch.FloatTensor')
end

function Module:double()
   return self:type('torch.DoubleTensor')
end

function Module:cuda()
   return self:type('torch.CudaTensor')
end

function Module:reset()
end

function Module:getParameters()
   -- get parameters
   local parameters,gradParameters = self:parameters()

   local function storageInSet(set, storage) --this is waste of time (need correct hash)
      for key, val in pairs(set) do
         if key == storage then
            return val
         end
      end
   end

   -- this function flattens arbitrary lists of parameters,
   -- even complex shared ones
   local function flatten(parameters)
      local storages = {}
      local nParameters = 0
      for k = 1,#parameters do
         if not storageInSet(storages, parameters[k]:storage()) then
            storages[parameters[k]:storage()] = nParameters
            nParameters = nParameters + parameters[k]:storage():size()
         end
      end
      
      local flatParameters = torch.Tensor(nParameters):fill(1)
      local flatStorage = flatParameters:storage()

      for k = 1,#parameters do
         local storageOffset = storageInSet(storages, parameters[k]:storage())
         parameters[k]:set(flatStorage,
                           storageOffset + parameters[k]:storageOffset(),
                           parameters[k]:size(),
                           parameters[k]:stride())
         parameters[k]:zero()
      end
      if (flatParameters:sum() ~= 0) then
         print("<getParameters()> WARNING: found " 
               .. flatParameters:sum() .. " holes in the parameters vector (i.e. "
               .. flatParameters:sum() .. " storage elements that are unused, this "
               .. "might be an issue for your optimization procedure)")
      end

      for k, v in pairs(storages) do
         flatParameters[{{v+1,v+k:size()}}]:copy(torch.Tensor():set(k))
      end
      return flatParameters
   end

   -- flatten parameters and gradients
   local flatParameters = flatten(parameters)
   local flatGradParameters = flatten(gradParameters)

   -- return new flat vector that contains all discrete parameters
   return flatParameters, flatGradParameters
end

function Module:__call__(input, gradOutput)
   self:forward(input)
   if gradOutput then
      self:backward(input, gradOutput)
      return self.output, self.gradInput
   else
      return self.output
   end
end
